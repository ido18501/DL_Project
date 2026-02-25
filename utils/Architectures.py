# utils/Architectures.py
# LOS++ (Movies-robust): peak-aware pooling + gated fusion (spectrum vs signature)
# Keeps API:
#   get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape)
#   model.forward(sorted_TDS_normalized, normalized_ATP, ATP_R) -> logits [B]
#
# NOTE: Your training pipeline uses BCEWithLogitsLoss and applies sigmoid for AUC.
# Therefore we return LOGITS (not probabilities).

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from utils.constants import MODEL_VOCAB_SIZES


# -----------------------------
# Helpers
# -----------------------------
def _get_attr(obj, name, default):
    return getattr(obj, name, default)

def _delta_leftpad(x: torch.Tensor) -> torch.Tensor:
    """delta(x) = concat(zeros_like(x[:,:1]), x[:,1:] - x[:,:-1], dim=1) for x [B,N,C]."""
    z = torch.zeros_like(x[:, :1])
    return torch.cat([z, x[:, 1:] - x[:, :-1]], dim=1)

def _top_p_mean(h: torch.Tensor, scores: torch.Tensor, p: float) -> torch.Tensor:
    """
    h:      [B, N, D]
    scores: [B, N]  (higher = more suspicious)
    p: fraction in (0,1], e.g. 0.10 means top 10% tokens
    returns: [B, D]
    """
    B, N, D = h.shape
    p = float(max(min(p, 1.0), 1e-6))
    k = max(1, int(round(p * N)))
    # topk indices by score
    topv, topi = torch.topk(scores, k=k, dim=1, largest=True, sorted=False)  # [B,k]
    idx = topi.unsqueeze(-1).expand(-1, -1, D)                                # [B,k,D]
    h_top = torch.gather(h, dim=1, index=idx)                                  # [B,k,D]
    return h_top.mean(dim=1)                                                   # [B,D]


# -----------------------------
# Depthwise-separable Conv block
# -----------------------------
class DepthwiseSeparableConv1DBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.pw = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            bias=True,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        residual = x
        y = x.transpose(1, 2)        # [B, D, N]
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)        # [B, N, D]
        y = self.act(y)
        y = self.drop(y)
        return residual + y


# -----------------------------
# Attn pooling (small MLP attention)
# -----------------------------
class AttnPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        h = max(8, d_model // 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        scores = self.mlp(x)                      # [B, N, 1]
        w = torch.softmax(scores, dim=1)          # [B, N, 1]
        return torch.sum(w * x, dim=1)            # [B, D]


# -----------------------------
# LOS++ (Movies robust)
# -----------------------------
class LOS_PP_MultiScaleDeltaTransformer(nn.Module):
    """
    LOS++ Multi-Scale Delta Transformer (Movies-robust upgrades)

    Forward:
        (sorted_TDS_normalized [B,N,V], normalized_ATP [B,N,1], ATP_R [B,N]) -> logits [B]
    """

    def __init__(self, args, max_sequence_length, input_dim=1):
        super().__init__()
        self.args = args
        self.max_sequence_length = int(max_sequence_length)

        # -----------------------
        # Core hparams
        # -----------------------
        self.k = int(_get_attr(args, "k", 20))
        self.top_p = float(_get_attr(args, "top_p", 0.10))
        self.conv_blocks = int(_get_attr(args, "conv_blocks", 2))
        self.transformer_layers = int(_get_attr(args, "transformer_layers", _get_attr(args, "num_layers", 1)))
        self.d_model = int(_get_attr(args, "hidden_dim", 128))
        self.dropout = float(_get_attr(args, "dropout", 0.1))

        self.conv_blocks = int(_get_attr(args, "conv_blocks", 2))
        self.transformer_layers = int(_get_attr(args, "transformer_layers", _get_attr(args, "num_layers", 1)))
        self.heads = int(_get_attr(args, "heads", 4))
        assert self.d_model % self.heads == 0, "hidden_dim must be divisible by heads"
        self.eps = 1e-9

        # -----------------------
        # Pooling
        # -----------------------
        # Supported:
        # - mean
        # - attn
        # - cls
        # - meanmaxcls
        # - toppmean        (mean of top-p suspicious tokens)
        # - meanmaxclstop   (concat [CLS, mean, max, top-p-mean])
        self.pooling = _get_attr(args, "pooling", None)
        if self.pooling is None:
            self.pooling = _get_attr(args, "pool", "attn")
        self.pooling = str(self.pooling).lower()

        self.top_p = float(_get_attr(args, "top_p", 0.10))  # for top-p pooling

        valid = {"mean", "attn", "cls", "meanmaxcls", "toppmean", "meanmaxclstop"}
        if self.pooling not in valid:
            raise ValueError(f"Invalid pooling='{self.pooling}'. Choose one of {sorted(valid)}")

        self.use_cls_token = self.pooling in {"cls", "meanmaxcls", "meanmaxclstop"}
        self.attn_pool = AttnPooling(self.d_model, self.dropout)

        # -----------------------
        # Rank encoding (LOS-Net style)
        # -----------------------
        self.rank_encoding = _get_attr(args, "rank_encoding", "scale_encoding")
        if self.rank_encoding not in {"scale_encoding", "one_hot_encoding"}:
            raise ValueError("rank_encoding must be 'scale_encoding' or 'one_hot_encoding'")

        self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.d_model // 4))
        if self.rank_encoding == "one_hot_encoding":
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.d_model // 4)

        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.d_model // 4))

        # -----------------------
        # Signature features (compressed)
        # -----------------------
        # logp_top(k) + entropy + margin + log_gap + p_tail + cdf2/5/10/20
        # + log_raw_atp + normalized_ATP + d_atp + d2_atp
        # + rank + d_rank + d_entropy + d2_entropy
        self.F = (
            self.k +          # logp_top
            1 + 1 + 1 + 1 +   # entropy, margin, log_gap, p_tail
            4 +               # cdf2,cdf5,cdf10,cdf20
            1 +               # log_raw_atp
            1 + 1 + 1 +       # normalized_ATP, d_atp, d2_atp
            1 + 1 +           # rank, d_rank
            1 + 1             # d_entropy, d2_entropy
        )
        self.logp_ln = nn.LayerNorm(self.k)

        # -----------------------
        # Spectrum branch (high-bandwidth)
        # -----------------------
        self.spectrum_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # Signature projection to D/2
        self.sig_proj = nn.Sequential(
            nn.Linear(self.F + (self.d_model // 4) + (self.d_model // 4), self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # -----------------------
        # NEW: gated fusion (spectrum vs signature)
        # -----------------------
        self.s2d = nn.Linear(self.d_model // 2, self.d_model, bias=False)
        self.t2d = nn.Linear(self.d_model // 2, self.d_model, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_model, max(16, self.d_model // 4)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(max(16, self.d_model // 4), 1),
        )
        self.fuse_ln = nn.LayerNorm(self.d_model)

        # -----------------------
        # CLS + positional embeddings
        # -----------------------
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.d_model)

        # -----------------------
        # Local conv blocks
        # -----------------------
        kernels = _get_attr(args, "conv_kernels", [3, 5, 7])
        if not isinstance(kernels, (list, tuple)) or len(kernels) == 0:
            kernels = [3, 5, 7]
        self.conv_blocks_list = nn.ModuleList([
            DepthwiseSeparableConv1DBlock(self.d_model, int(kernels[i % len(kernels)]), self.dropout)
            for i in range(max(self.conv_blocks, 0))
        ])

        # -----------------------
        # Transformer
        # -----------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.heads,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.transformer_layers)

        # -----------------------
        # NEW: token scoring head for peak-aware pooling
        # -----------------------
        self.token_scorer = nn.Linear(self.d_model, 1)

        # Pool output dim
        if self.pooling in {"mean", "attn", "cls", "toppmean"}:
            self.pool_out_dim = self.d_model
        elif self.pooling == "meanmaxcls":
            self.pool_out_dim = 3 * self.d_model
        elif self.pooling == "meanmaxclstop":
            self.pool_out_dim = 4 * self.d_model
        else:
            raise RuntimeError("Unhandled pooling mode")

        # Classification head: -> 1 (logit)
        self.seq_head = nn.Sequential(
            nn.Linear(self.pool_out_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        """
        Inputs:
            sorted_TDS_normalized: [B, N, V] (descending over V)
            normalized_ATP:        [B, N, 1]
            ATP_R:                [B, N]
        Output:
            logits: [B]
        """
        B, N, V = sorted_TDS_normalized.shape
        device = sorted_TDS_normalized.device

        # -----------------------
        # Spectrum branch
        # -----------------------
        spectrum = self.spectrum_proj(sorted_TDS_normalized.to(torch.float32))  # [B,N,D/2]

        # -----------------------
        # Signature features
        # -----------------------
        k = min(self.k, V)
        p_top = sorted_TDS_normalized[:, :, :k].to(torch.float32)
        p_top = torch.clamp(p_top, min=0.0)
        p_tail = 1.0 - torch.sum(p_top, dim=-1, keepdim=True)
        p_tail = torch.clamp(p_tail, min=0.0)

        logp_top = torch.log(p_top + self.eps)
        logp_top = torch.nan_to_num(logp_top, nan=0.0, posinf=0.0, neginf=0.0)

        entropy_top = -torch.sum(p_top * logp_top, dim=-1, keepdim=True)  # [B,N,1]

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
        cdf20 = torch.sum(p_top, dim=-1, keepdim=True)  # top-k mass

        normalized_ATP = normalized_ATP.to(torch.float32)
        raw_ATP = normalized_ATP  # placeholder (you said upstream raw ATP not provided)
        log_raw_atp = torch.log(raw_ATP + self.eps)

        d_atp = _delta_leftpad(normalized_ATP)
        d2_atp = _delta_leftpad(d_atp)

        d_entropy = _delta_leftpad(entropy_top)
        d2_entropy = _delta_leftpad(d_entropy)

        rank = ATP_R.unsqueeze(-1).to(torch.float32)
        d_rank = _delta_leftpad(rank)

        # pad logp_top to self.k
        if k < self.k:
            pad = torch.zeros(B, N, self.k - k, device=device, dtype=logp_top.dtype)
            logp_top_full = torch.cat([logp_top, pad], dim=-1)
        else:
            logp_top_full = logp_top

        logp_top_full = self.logp_ln(logp_top_full)

        base_feats = torch.cat([
            logp_top_full,
            entropy_top, margin, log_gap, p_tail,
            cdf2, cdf5, cdf10, cdf20,
            log_raw_atp,
            normalized_ATP, d_atp, d2_atp,
            rank, d_rank,
            d_entropy, d2_entropy,
        ], dim=-1)
        base_feats = torch.nan_to_num(base_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # -----------------------
        # Rank + ATP encodings
        # -----------------------
        if self.rank_encoding == "scale_encoding":
            vocab = float(MODEL_VOCAB_SIZES[self.args.LLM])
            encoded_ATP_R = 2.0 * (0.5 - (ATP_R.to(torch.float32) / vocab))  # [B,N]
            encoded_ATP_R = normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
        else:
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)

        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP

        sig_in = torch.cat([base_feats, encoded_ATP_R, encoded_normalized_ATP], dim=-1)
        sig = self.sig_proj(sig_in)  # [B,N,D/2]

        # -----------------------
        # NEW: gated fusion into D
        # -----------------------
        sD = self.s2d(spectrum)   # [B,N,D]
        tD = self.t2d(sig)        # [B,N,D]
        g = torch.sigmoid(self.gate_mlp(self.fuse_ln(sD + tD)))  # [B,N,1]
        x = self.fuse_ln(g * sD + (1.0 - g) * tD)                # [B,N,D]

        # -----------------------
        # CLS + positional embeddings
        # -----------------------
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)
            x = torch.cat([cls, x], dim=1)  # [B,1+N,D]
            seq_len = x.size(1)
            pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
            x = x + self.pos_embedding(pos_idx)
        else:
            pos_idx = torch.arange(N, device=device).unsqueeze(0)
            x = x + self.pos_embedding(pos_idx)

        # -----------------------
        # Local conv
        # -----------------------
        for blk in self.conv_blocks_list:
            x = blk(x)

        # -----------------------
        # Transformer
        # -----------------------
        h = self.transformer(x)  # [B,seq_len,D]

        # -----------------------
        # Pooling
        # -----------------------
        if self.use_cls_token:
            cls_vec = h[:, 0, :]          # [B,D]
            tok = h[:, 1:, :]             # [B,N,D]
        else:
            cls_vec = None
            tok = h                        # [B,N,D]

        if self.pooling == "mean":
            pooled = tok.mean(dim=1)

        elif self.pooling == "attn":
            pooled = self.attn_pool(tok)

        elif self.pooling == "cls":
            pooled = cls_vec

        elif self.pooling == "meanmaxcls":
            tok_mean = tok.mean(dim=1)
            tok_max = tok.max(dim=1).values
            pooled = torch.cat([cls_vec, tok_mean, tok_max], dim=-1)

        elif self.pooling == "toppmean":
            # score tokens, then mean top-p by score
            tok_scores = self.token_scorer(tok).squeeze(-1)      # [B,N]
            pooled = _top_p_mean(tok, tok_scores, self.top_p)

        elif self.pooling == "meanmaxclstop":
            tok_scores = self.token_scorer(tok).squeeze(-1)      # [B,N]
            tok_top = _top_p_mean(tok, tok_scores, self.top_p)
            tok_mean = tok.mean(dim=1)
            tok_max = tok.max(dim=1).values
            pooled = torch.cat([cls_vec, tok_mean, tok_max, tok_top], dim=-1)

        else:
            raise RuntimeError(f"Unhandled pooling mode: {self.pooling}")

        # -----------------------
        # Head (logits)
        # -----------------------
        logits = self.seq_head(pooled).squeeze(-1)  # [B]
        return logits


# -----------------------------
# get_model mapping
# -----------------------------
def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    model_mapping = {
        "LOS++": LOS_PP_MultiScaleDeltaTransformer,
    }

    if args.probe_model not in model_mapping:
        raise ValueError(
            f"Unknown/unsupported model in this file: {args.probe_model}. "
            f"Supported: {sorted(model_mapping.keys())}"
        )

    return model_mapping[args.probe_model](
        args=args,
        max_sequence_length=max_sequence_length,
        input_dim=input_dim
    )