# utils/Architectures.py
# LOS++ with: group-wise normalization (preserve calibration scale),
# real CLS token support, and pooling modes including "meanmaxcls"
# (concat [CLS, mean(tokens), max(tokens)]).

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import MODEL_VOCAB_SIZES
from einops import repeat

# -----------------------------
# Helpers
# -----------------------------
def _get_attr(obj, name, default):
    return getattr(obj, name, default)

def _delta_leftpad(x: torch.Tensor) -> torch.Tensor:
    """
    delta(x) = concat(zeros_like(x[:,:1]), x[:,1:] - x[:,:-1], dim=1)
    Works for x shape [B, N, C]
    """
    z = torch.zeros_like(x[:, :1])
    return torch.cat([z, x[:, 1:] - x[:, :-1]], dim=1)

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
# Attention pooling
# -----------------------------
class AttnPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.last_weights = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]
        scores = self.mlp(h)                 # [B, N, 1]
        w = torch.softmax(scores, dim=1)     # [B, N, 1]
        self.last_weights = w.detach()
        pooled = torch.sum(w * h, dim=1)     # [B, D]
        return pooled

# -----------------------------
# LOS++
# -----------------------------
class LOS_PP_MultiScaleDeltaTransformer(nn.Module):
    """
    LOS++ Multi-Scale Delta Transformer

    Forward signature MUST match existing code:
        forward(sorted_TDS_normalized, normalized_ATP, ATP_R) -> [B]
    """

    def __init__(self, args, max_sequence_length, input_dim=1):
        super().__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length

        # -----------------------
        # Core hparams
        # -----------------------
        self.k = _get_attr(args, "k", 20)  # used for LOS++ signature stats
        self.d_model = _get_attr(args, "hidden_dim", 128)
        self.dropout = _get_attr(args, "dropout", 0.1)
        self.conv_blocks = _get_attr(args, "conv_blocks", 3)
        self.transformer_layers = _get_attr(args, "transformer_layers", _get_attr(args, "num_layers", 2))
        self.heads = _get_attr(args, "heads", 4)

        # Pooling options:
        # - mean: mean over tokens
        # - attn: attention pooling over tokens
        # - cls: CLS only
        # - meanmaxcls: concat [CLS, mean(tokens), max(tokens)] -> 3*d_model
        self.pooling = _get_attr(args, "pooling", None)
        if self.pooling is None:
            self.pooling = _get_attr(args, "pool", "attn")
        self.pooling = str(self.pooling).lower()
        valid = {"mean", "attn", "cls", "meanmaxcls"}
        if self.pooling not in valid:
            raise ValueError(f"Invalid pooling='{self.pooling}'. Choose one of {sorted(valid)}")

        self.use_cls_token = self.pooling in {"cls", "meanmaxcls"}
        assert self.d_model % self.heads == 0, "d_model must be divisible by nhead"
        self.eps = 1e-9

        # -----------------------
        # Rank encoding: LOS-Net-style scale encoding
        # -----------------------
        # We keep this compatible with your args; if absent default to scale_encoding.
        self.rank_encoding = _get_attr(args, "rank_encoding", "scale_encoding")
        if self.rank_encoding not in {"scale_encoding", "one_hot_encoding"}:
            raise ValueError("rank_encoding must be 'scale_encoding' or 'one_hot_encoding'")

        # scale encoding parameter (LOS-Net style)
        self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.d_model // 4))

        # optional one-hot rank embedding (very high-cardinality; usually not recommended)
        if self.rank_encoding == "one_hot_encoding":
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.d_model // 4)

        # normalized ATP parameter
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.d_model // 4))

        # -----------------------
        # LOS++ signature features
        # (compressed, robust cues)
        # -----------------------
        # logp_top(k) + entropy + margin + log_gap + p_tail + cdf2/5/10/20
        # + log_raw_atp (placeholder) + normalized_ATP + d_atp + d2_atp
        # + rank + d_rank + d_entropy + d2_entropy
        # + (rank_encoding vector already injected separately here as part of sig block)
        #
        # NOTE: we don't use rank buckets anymore in this hybrid; we use scale encoding.
        self.F = (
                self.k +  # logp_top
                1 + 1 + 1 + 1 +  # entropy, margin, log_gap, p_tail
                4 +  # cdf2,cdf5,cdf10,cdf20
                1 +  # log_raw_atp (placeholder)
                1 + 1 + 1 +  # normalized_ATP, d_atp, d2_atp
                1 + 1 +  # rank, d_rank
                1 + 1  # d_entropy, d2_entropy
        )

        # Normalize only logp_top block to preserve scalar calibration
        self.logp_ln = nn.LayerNorm(self.k)

        # Optional stochastic feature masking (training only)
        self.feature_masking = _get_attr(args, "feature_masking", False)
        self.feature_mask_p = _get_attr(args, "feature_mask_p", 0.1)
        self.feature_drop = nn.Dropout(self.dropout)

        # -----------------------
        # Hybrid: spectrum branch (LOS-Net style)
        # This is the missing high-bandwidth path for Movies.
        # -----------------------
        # input_dim here is topk_dim from dataset (e.g., 1000)
        self.spectrum_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # -----------------------
        # Signature branch projection to D/2
        # -----------------------
        self.sig_proj = nn.Sequential(
            nn.Linear(self.F + (self.d_model // 4) + (self.d_model // 4), self.d_model // 2),
            #            ^ base features  ^ rank-enc vec          ^ norm_ATP vec
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # -----------------------
        # Optional learned CLS + positional embeddings (LOS-Net bias)
        # -----------------------
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.d_model)

        # -----------------------
        # Local conv blocks (optional)
        # -----------------------
        kernels = _get_attr(args, "conv_kernels", [3, 5, 7])
        if not isinstance(kernels, (list, tuple)) or len(kernels) == 0:
            kernels = [3, 5, 7]
        self.conv_blocks_list = nn.ModuleList([
            DepthwiseSeparableConv1DBlock(
                d_model=self.d_model,
                kernel_size=int(kernels[i % len(kernels)]),
                dropout=self.dropout
            )
            for i in range(max(int(self.conv_blocks), 0))
        ])

        # -----------------------
        # Global transformer
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

        # Pooling module (for "attn")
        self.attn_pool = AttnPooling(self.d_model, self.dropout)

        # Pool output dim
        if self.pooling in {"mean", "attn", "cls"}:
            self.pool_out_dim = self.d_model
        elif self.pooling == "meanmaxcls":
            self.pool_out_dim = 3 * self.d_model
        else:
            raise RuntimeError("Unhandled pooling mode")

        # Classification head: pool_out_dim -> d_model -> 1 (logits)
        self.seq_head = nn.Sequential(
            nn.Linear(self.pool_out_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        # Optional token head (OFF by default)
        self.return_token_scores = _get_attr(args, "return_token_scores", False)
        if self.return_token_scores:
            self.token_head = nn.Linear(self.d_model, 1)

    def _rank_bucket_ids(self, ATP_R: torch.Tensor) -> torch.Tensor:
        r = ATP_R.long().clamp_min(0)
        bucket = torch.bucketize(r, self.rank_bucket_thresholds, right=True)
        return bucket.clamp_max(self.rank_bucket_emb.num_embeddings - 1)

    def _apply_group_feature_mask(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Groups:
        - logp block
        - shape stats + d_entropy/d2_entropy
        - ATP block (log_raw_atp, normalized_ATP, d_atp, d2_atp)
        - rank + rank_emb
        """
        if (not self.training) or (not self.feature_masking) or (self.feature_mask_p <= 0):
            return feats

        B, N, _ = feats.shape
        device = feats.device

        i = 0
        sl_logp = slice(i, i + self.k); i += self.k
        sl_shape = slice(i, i + (1 + 1 + 1 + 1 + 4)); i += (1 + 1 + 1 + 1 + 4)
        sl_atp = slice(i, i + 4); i += 4
        sl_rank = slice(i, i + 2); i += 2
        sl_dent = slice(i, i + 2); i += 2
        sl_rankemb = slice(i, i + self.rank_emb_dim); i += self.rank_emb_dim

        groups = [
            [sl_logp],
            [sl_shape, sl_dent],
            [sl_atp],
            [sl_rank, sl_rankemb],
        ]

        for g in groups:
            drop = (torch.rand(B, 1, 1, device=device) < self.feature_mask_p).float()
            if drop.max() == 0:
                continue
            for sl in g:
                feats[:, :, sl] = feats[:, :, sl] * (1.0 - drop)

        return feats

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        """
        Inputs:
            sorted_TDS_normalized: [B, N, V]  (descending over V)   (V==topk_dim)
            normalized_ATP:        [B, N, 1]
            ATP_R:                [B, N]

        Output:
            logits: [B]
        """
        B, N, V = sorted_TDS_normalized.shape
        device = sorted_TDS_normalized.device

        # -----------------------
        # Spectrum branch (LOS-Net style): embed full topk_dim vector
        # -----------------------
        spectrum = self.spectrum_proj(sorted_TDS_normalized.to(torch.float32))  # [B,N,D/2]

        # -----------------------
        # LOS++ signature features (compressed, robust)
        # -----------------------
        k = min(self.k, V)

        p_top = sorted_TDS_normalized[:, :, :k].to(torch.float32)  # [B,N,k]
        p_top = torch.clamp(p_top, min=0.0)
        p_tail = 1.0 - torch.sum(p_top, dim=-1, keepdim=True)  # [B,N,1]
        p_tail = torch.clamp(p_tail, min=0.0)

        logp_top = torch.log(p_top + self.eps)
        logp_top = torch.nan_to_num(logp_top, nan=0.0, posinf=0.0, neginf=0.0)

        entropy_top = -torch.sum(p_top * logp_top, dim=-1, keepdim=True)  # [B,N,1]

        if k >= 2:
            margin = p_top[..., 0:1] - p_top[..., 1:2]  # [B,N,1]
            log_gap = logp_top[..., 0:1] - logp_top[..., 1:2]  # [B,N,1]
            cdf2 = torch.sum(p_top[..., :2], dim=-1, keepdim=True)  # [B,N,1]
        else:
            margin = torch.zeros(B, N, 1, device=device)
            log_gap = torch.zeros_like(margin)
            cdf2 = torch.sum(p_top, dim=-1, keepdim=True)

        cdf5 = torch.sum(p_top[..., :min(5, k)], dim=-1, keepdim=True)
        cdf10 = torch.sum(p_top[..., :min(10, k)], dim=-1, keepdim=True)
        cdf20 = torch.sum(p_top, dim=-1, keepdim=True)  # top-k mass

        # normalized_ATP dynamics
        normalized_ATP = normalized_ATP.to(torch.float32)
        raw_ATP = normalized_ATP  # placeholder until you provide raw ATP
        log_raw_atp = torch.log(raw_ATP + self.eps)

        d_atp = _delta_leftpad(normalized_ATP)
        d2_atp = _delta_leftpad(d_atp)

        d_entropy = _delta_leftpad(entropy_top)
        d2_entropy = _delta_leftpad(d_entropy)

        # Rank scalar + delta
        rank = ATP_R.unsqueeze(-1).to(torch.float32)  # [B,N,1]
        d_rank = _delta_leftpad(rank)

        # Pad logp_top to self.k
        if k < self.k:
            pad = torch.zeros(B, N, self.k - k, device=device, dtype=logp_top.dtype)
            logp_top_full = torch.cat([logp_top, pad], dim=-1)  # [B,N,self.k]
        else:
            logp_top_full = logp_top

        # Normalize only logp block
        logp_top_full = self.logp_ln(logp_top_full)

        base_feats = torch.cat([
            logp_top_full,  # [B,N,k]
            entropy_top,  # [B,N,1]
            margin,  # [B,N,1]
            log_gap,  # [B,N,1]
            p_tail,  # [B,N,1]
            cdf2, cdf5, cdf10, cdf20,  # [B,N,4]
            log_raw_atp,  # [B,N,1]
            normalized_ATP,  # [B,N,1]
            d_atp, d2_atp,  # [B,N,2]
            rank, d_rank,  # [B,N,2]
            d_entropy, d2_entropy,  # [B,N,2]
        ], dim=-1)  # [B,N,F]

        base_feats = torch.nan_to_num(base_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature masking (optional): very light regularizer against shortcuts
        if self.training and self.feature_masking and self.feature_mask_p > 0:
            # Mask base feature groups per batch (same mask for all tokens)
            drop = (torch.rand(B, 1, 1, device=device) < self.feature_mask_p).float()
            # only mask the base_feats (not spectrum)
            base_feats = base_feats * (1.0 - drop)

        base_feats = self.feature_drop(base_feats)

        # -----------------------
        # Rank + ATP encodings (LOS-Net style)
        # -----------------------
        # encoded_ATP_R = normalized_ATP * (2*(0.5 - R/Vocab)) * param
        if self.rank_encoding == "scale_encoding":
            vocab = float(MODEL_VOCAB_SIZES[self.args.LLM])
            encoded_ATP_R = 2.0 * (0.5 - (ATP_R.to(torch.float32) / vocab))  # [B,N]
            encoded_ATP_R = normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R  # [B,N,D/4]
        else:
            # one-hot embedding path (heavy)
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)  # [B,N,D/4]

        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP  # [B,N,D/4]

        # Signature embedding to D/2
        sig_in = torch.cat([base_feats, encoded_ATP_R, encoded_normalized_ATP], dim=-1)  # [B,N,F + D/4 + D/4]
        sig = self.sig_proj(sig_in)  # [B,N,D/2]

        # -----------------------
        # Fuse spectrum + signature -> D
        # -----------------------
        x = torch.cat([spectrum, sig], dim=-1)  # [B,N,D]

        # -----------------------
        # Add CLS + positional embeddings (LOS-Net bias)
        # -----------------------
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)  # [B,1,D]
            x = torch.cat([cls, x], dim=1)  # [B,1+N,D]
            seq_len = x.size(1)
            pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
            x = x + self.pos_embedding(pos_idx)
        else:
            pos_idx = torch.arange(N, device=device).unsqueeze(0)
            x = x + self.pos_embedding(pos_idx)

        # -----------------------
        # Optional local conv
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
        if self.pooling == "mean":
            if self.use_cls_token:
                pooled = h[:, 1:, :].mean(dim=1)
            else:
                pooled = h.mean(dim=1)

        elif self.pooling == "attn":
            if self.use_cls_token:
                pooled = self.attn_pool(h[:, 1:, :])
            else:
                pooled = self.attn_pool(h)

        elif self.pooling == "cls":
            pooled = h[:, 0, :]  # CLS

        elif self.pooling == "meanmaxcls":
            cls_vec = h[:, 0, :]  # [B,D]
            tok = h[:, 1:, :]  # [B,N,D]
            tok_mean = tok.mean(dim=1)  # [B,D]
            tok_max = tok.max(dim=1).values  # [B,D]
            pooled = torch.cat([cls_vec, tok_mean, tok_max], dim=-1)  # [B,3D]

        else:
            raise RuntimeError(f"Unhandled pooling mode: {self.pooling}")

        # -----------------------
        # Head (logits)
        # -----------------------
        logits = self.seq_head(pooled).squeeze(-1)  # [B]

        if self.return_token_scores:
            token_logits = self.token_head(h)  # [B,seq_len,1]
            return logits, token_logits

        return logits


# -----------------------------
# get_model mapping
# -----------------------------
def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    model_mapping = {
        'LOS++': LOS_PP_MultiScaleDeltaTransformer,
    }

    if args.probe_model in {'LOS-Net', 'LOS++', 'ATP_R_Transf'}:
        return model_mapping[args.probe_model](args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)
    elif args.probe_model in {'ATP_R_MLP'}:
        return model_mapping[args.probe_model](args=args, actual_sequence_length=actual_sequence_length)
    else:
        raise ValueError(f"Unknown model: {args.probe_model}")