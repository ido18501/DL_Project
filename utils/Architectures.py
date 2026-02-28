import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from utils.constants import MODEL_VOCAB_SIZES





def delta_left_pad(x):
    # here we compute first order derivative with left padding
    z = torch.zeros_like(x[:, :1])
    return torch.cat([z, x[:, 1:] - x[:, :-1]], dim=1)

def top_p_mean(h, scores, p):
    """
    h:      [B, N, D]
    scores: [B, N]  (higher means more suspicion)
    p: value in (0,1] , corresponds to a percentage
    returns: [B, D]
    """
    B, N, D = h.shape
    # min and max guarding for safe operations
    p = float(max(min(p, 1.0), 1e-6))
    k = max(1, int(round(p * N)))
    # top-k indices by score
    top_v, top_i = torch.topk(scores, k=k, dim=1, largest=True, sorted=False)
    idx = top_i.unsqueeze(-1).expand(-1, -1, D)
    z = torch.gather(h, dim=1, index=idx)
    return z.mean(dim=1)


# convolution block to apply before transformer
class DepthwiseSeparableConv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size, dropout):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.pw = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            bias=True,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        y = x.transpose(1, 2)
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)
        y = self.act(y)
        y = self.drop(y)
        return residual + y


# attention pooling class
class AttnPooling(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        h = max(8, dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, x):

        scores = self.mlp(x)
        w = torch.softmax(scores, dim=1)
        return torch.sum(w * x, dim=1)


# LOS++ class implementation
class LOS_PP_MultiScaleDeltaTransformer(nn.Module):

    def __init__(self, args, max_sequence_length, input_dim=1):
        super().__init__()
        self.args = args
        self.max_sequence_length = int(max_sequence_length)

        # hp's
        self.k = 20
        self.top_p = float(getattr(args, "top_p", 0.10))
        self.conv_blocks = int(getattr(args, "conv_blocks", 2))
        self.transformer_layers = int(getattr(args, "transformer_layers", getattr(args, "num_layers", 1)))
        self.dim = int(getattr(args, "hidden_dim", 128))
        self.dropout = float(getattr(args, "dropout", 0.15))
        self.transformer_layers = int(getattr(args, "transformer_layers", getattr(args, "num_layers", 1)))
        self.heads = int(getattr(args, "heads", 4))
        assert self.dim % self.heads == 0, "hidden_dim must be divisible by number of heads"
        # prevent division by 0
        self.eps = 1e-9
        # pooling options
        self.pooling = getattr(args, "pool", "meanmaxcls")
        self.pooling = str(self.pooling).lower()

        self.top_p = float(getattr(args, "top_p", 0.10))  # for top-p pooling

        if self.pooling not in {"mean", "attn", "cls", "meanmaxcls", "toppmean", "meanmaxclstop"}:
            raise ValueError(f"Invalid pooling")

        self.use_cls_token = self.pooling in {"cls", "meanmaxcls", "meanmaxclstop"}
        self.attn_pool = AttnPooling(self.dim, self.dropout)

        # Rank encoding
        self.rank_encoding = getattr(args, "rank_encoding", "scale_encoding")
        if self.rank_encoding not in {"scale_encoding", "one_hot_encoding"}:
            raise ValueError("Invalid rank encoding")

        self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.dim // 4))
        if self.rank_encoding == "one_hot_encoding":
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.dim // 4)

        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.dim // 4))
        # The uneffective way of calculation below was meant to make each parameter's rule clear
        self.F = (
            self.k +          # log(p_top)
            1 + 1 + 1 + 1 +   # entropy, margin, log_gap, p_tail
            4 +               # cdf2,cdf5,cdf10,cdfk
            1 +               # log_raw_atp
            1 + 1 + 1 +       # normalized_ATP, d_atp, d2_atp
            1 + 1 +           # rank, d_rank
            1 + 1             # d_entropy, d2_entropy
        )
        self.logp_ln = nn.LayerNorm(self.k)

        # spectrum branch
        self.spectrum_proj = nn.Sequential(
            nn.Linear(input_dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # Signature projection to D/2
        self.sig_proj = nn.Sequential(
            nn.Linear(self.F + (self.dim // 4) + (self.dim // 4), self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

       # gated fusion
        self.s2d = nn.Linear(self.dim // 2, self.dim, bias=False)
        self.t2d = nn.Linear(self.dim // 2, self.dim, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.dim, max(16, self.dim // 4)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(max(16, self.dim // 4), 1),
        )
        self.fuse_ln = nn.LayerNorm(self.dim)

       # cls and positional embeddings
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.dim)

        # local convolution vlocks- should be applied before transformer
        kernels = getattr(args, "conv_kernels", [3, 5, 7])
        if not isinstance(kernels, (list, tuple)) or len(kernels) == 0:
            kernels = [3, 5, 7]
        self.conv_blocks_list = nn.ModuleList([
            DepthwiseSeparableConv1DBlock(self.dim, int(kernels[i % len(kernels)]), self.dropout)
            for i in range(max(self.conv_blocks, 0))
        ])

       # transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.heads,
            dim_feedforward=self.dim * 2,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.transformer_layers)

        # score for peak aware pooling - further testing should be done for this one
        self.token_scorer = nn.Linear(self.dim, 1)

        # Pool output dim - each method requires different dim
        if self.pooling in {"mean", "attn", "cls", "toppmean"}:
            self.pool_out_dim = self.dim
        elif self.pooling == "meanmaxcls":
            self.pool_out_dim = 3 * self.dim
        elif self.pooling == "meanmaxclstop":
            self.pool_out_dim = 4 * self.dim
        else:
            raise RuntimeError("Unhandled pooling mode")

        # classification head
        self.seq_head = nn.Sequential(
            nn.Linear(self.pool_out_dim, self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, 1),
        )

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        B, N, V = sorted_TDS_normalized.shape
        device = sorted_TDS_normalized.device

        # spectrum brunch
        spectrum = self.spectrum_proj(sorted_TDS_normalized.to(torch.float32))

       # sig feature claculated below
        k = min(self.k, V)
        p_top = sorted_TDS_normalized[:, :, :k].to(torch.float32)
        p_top = torch.clamp(p_top, min=0.0)
        p_tail = 1.0 - torch.sum(p_top, dim=-1, keepdim=True)
        p_tail = torch.clamp(p_tail, min=0.0)

        logp_top = torch.log(p_top + self.eps)
        logp_top = torch.nan_to_num(logp_top, nan=0.0, posinf=0.0, neginf=0.0)

        entropy_top = -torch.sum(p_top * logp_top, dim=-1, keepdim=True)

        if k >= 2:
            margin = p_top[..., 0:1] - p_top[..., 1:2]
            log_gap = logp_top[..., 0:1] - logp_top[..., 1:2]
            cdf2 = torch.sum(p_top[..., :2], dim=-1, keepdim=True)
        else:
            margin = torch.zeros(B, N, 1, device=device)
            log_gap = torch.zeros_like(margin)
            cdf2 = torch.sum(p_top, dim=-1, keepdim=True)

        cdf5 = torch.sum(p_top[..., :min(5, k)], dim=-1, keepdim=True)
        cdf10 = torch.sum(p_top[..., :min(10, k)], dim=-1, keepdim=True)
        cdfk = torch.sum(p_top, dim=-1, keepdim=True)  # top-k mass

        normalized_ATP = normalized_ATP.to(torch.float32)
        log_raw_atp = torch.log(normalized_ATP  + self.eps)

        # first and second order derivatives for both entropy and atp
        d_atp = delta_left_pad(normalized_ATP)
        d2_atp = delta_left_pad(d_atp)

        d_entropy = delta_left_pad(entropy_top)
        d2_entropy = delta_left_pad(d_entropy)

        rank = ATP_R.unsqueeze(-1).to(torch.float32)
        d_rank = delta_left_pad(rank)

        # pad in case V<k
        if k < self.k:
            pad = torch.zeros(B, N, self.k - k, device=device, dtype=logp_top.dtype)
            logp_top_full = torch.cat([logp_top, pad], dim=-1)
        else:
            logp_top_full = logp_top

        logp_top_full = self.logp_ln(logp_top_full)

        base_feats = torch.cat([
            logp_top_full,
            entropy_top, margin, log_gap, p_tail,
            cdf2, cdf5, cdf10, cdfk,
            log_raw_atp,
            normalized_ATP, d_atp, d2_atp,
            rank, d_rank,
            d_entropy, d2_entropy,
        ], dim=-1)
        base_feats = torch.nan_to_num(base_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # encodings
        if self.rank_encoding == "scale_encoding":
            vocab = float(MODEL_VOCAB_SIZES[self.args.LLM])
            encoded_ATP_R = 2.0 * (0.5 - (ATP_R.to(torch.float32) / vocab))
            encoded_ATP_R = normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
        else:
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)

        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP

        sig_in = torch.cat([base_feats, encoded_ATP_R, encoded_normalized_ATP], dim=-1)
        sig = self.sig_proj(sig_in)

       # gated fusion
        sD = self.s2d(spectrum)
        tD = self.t2d(sig)
        g = torch.sigmoid(self.gate_mlp(self.fuse_ln(sD + tD)))
        x = self.fuse_ln(g * sD + (1.0 - g) * tD)

       # clas and position embedding
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.dim)
            x = torch.cat([cls, x], dim=1)
            seq_len = x.size(1)
            pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
            x = x + self.pos_embedding(pos_idx)
        else:
            pos_idx = torch.arange(N, device=device).unsqueeze(0)
            x = x + self.pos_embedding(pos_idx)

       # convolutions
        for blk in self.conv_blocks_list:
            x = blk(x)

        # transformer
        h = self.transformer(x)

        # pooling
        if self.use_cls_token:
            cls = h[:, 0, :]
            token = h[:, 1:, :]
        else:
            cls = None
            token = h

        if self.pooling == "mean":
            pooled = token.mean(dim=1)

        elif self.pooling == "attn":
            pooled = self.attn_pool(token)

        elif self.pooling == "cls":
            pooled = cls

        elif self.pooling == "meanmaxcls":
            pooled = torch.cat([cls, token.mean(dim=1), token.max(dim=1).values], dim=-1)

        # further experiments needed for this method
        elif self.pooling == "toppmean":
            pooled = top_p_mean(token, self.token_scorer(token).squeeze(-1), self.top_p)

        elif self.pooling == "meanmaxclstop":
            pooled = torch.cat([cls, token.mean(dim=1), token.max(dim=1).values, top_p_mean(token, self.token_scorer(token).squeeze(-1), self.top_p)], dim=-1)

        else:
            raise RuntimeError(f"Pooling not supported")

       # head - returns logits!
        logits = self.seq_head(pooled).squeeze(-1)
        return logits


# model mapping - currently only supports LOS++
def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    model_mapping = {"LOS++": LOS_PP_MultiScaleDeltaTransformer}

    if args.probe_model not in model_mapping:
        raise ValueError(
            f"Unsupported model."
            f"Supported: {sorted(model_mapping.keys())}"
        )

    return model_mapping[args.probe_model](
        args=args,
        max_sequence_length=max_sequence_length,
        input_dim=input_dim
    )