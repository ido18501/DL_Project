# utils/Architectures.py

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
    delta(x) = concat(zeros_like(x[:,:1]), x[:,1:] - x[:, :-1], dim=1)
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

        # Optional: stash last attention weights for diagnostics (not used by main.py)
        self.last_weights = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]
        scores = self.mlp(h)                 # [B, N, 1]
        w = torch.softmax(scores, dim=1)     # [B, N, 1]
        self.last_weights = w.detach()
        pooled = torch.sum(w * h, dim=1)     # [B, D]
        return pooled

# -----------------------------
# LOS++ Candidate
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

        # Defaults (overrideable by args if present)
        self.k = _get_attr(args, "k", 20)
        self.rank_emb_dim = _get_attr(args, "rank_emb_dim", 16)
        self.d_model = _get_attr(args, "hidden_dim", 128)
        self.dropout = _get_attr(args, "dropout", 0.1)
        self.conv_blocks = _get_attr(args, "conv_blocks", 3)

        # Backwards compat: many runs use --num_layers
        self.transformer_layers = _get_attr(args, "transformer_layers", _get_attr(args, "num_layers", 2))
        self.heads = _get_attr(args, "heads", 4)
        self.pooling = _get_attr(args, "pooling", None)

        # Pooling options:
        # - "mean": mean over tokens
        # - "attn": attention pooling over tokens
        # - "cls": use a learned CLS token (prepended)
        # - "minmaxcls": concat [CLS, min(tokens), max(tokens)]  -> 3*d_model
        if self.pooling is None:
            self.pooling = _get_attr(args, "pool", "attn")
        self.pooling = str(self.pooling).lower()

        valid = {"mean", "attn", "cls", "minmaxcls"}
        if self.pooling not in valid:
            raise ValueError(f"Invalid pooling='{self.pooling}'. Choose one of {sorted(valid)}")

        self.use_cls_token = self.pooling in {"cls", "minmaxcls"}

        assert self.d_model % self.heads == 0, "d_model must be divisible by nhead"
        self.eps = 1e-9

        # Rank buckets: [1,2,3,4,5,10,20,50,100,200,500,1000, inf]
        self.register_buffer(
            "rank_bucket_thresholds",
            torch.tensor([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=torch.long),
            persistent=False,
        )
        self.rank_bucket_emb = nn.Embedding(
            num_embeddings=len(self.rank_bucket_thresholds) + 1,  # +1 for >1000
            embedding_dim=self.rank_emb_dim,
        )

        # Feature dimension (UPDATED: added log_raw_atp)
        # concat order:
        # logp_top(k)
        # entropy(1), margin(1), log_gap(1), p_tail(1)
        # cdf2,cdf5,cdf10,cdf20 (4)
        # log_raw_atp(1), normalized_ATP(1), d_atp(1), d2_atp(1)
        # rank(1), d_rank(1)
        # d_entropy(1), d2_entropy(1)
        # rank_emb(rank_emb_dim)
        self.F = (
            self.k +                 # logp_top
            1 + 1 + 1 + 1 +          # entropy, margin, log_gap, p_tail
            4 +                      # cdf2, cdf5, cdf10, cdf20
            1 + 1 + 1 + 1 +          # log_raw_atp, normalized_ATP, d_atp, d2_atp
            1 + 1 +                  # rank, d_rank
            1 + 1 +                  # d_entropy, d2_entropy
            self.rank_emb_dim        # rank embedding
        )

        # Group-wise normalization:
        # normalize only logp_top block to preserve calibration scale of scalar features.
        self.logp_ln = nn.LayerNorm(self.k)
        self.feature_drop = nn.Dropout(self.dropout)

        # Optional stochastic feature masking (training only)
        self.feature_masking = _get_attr(args, "feature_masking", False)
        self.feature_mask_p = _get_attr(args, "feature_mask_p", 0.1)

        # Token projection: F -> d_model
        self.token_proj = nn.Sequential(
            nn.Linear(self.F, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # Local conv branch (multi-scale kernels cycling)
        kernels = _get_attr(args, "conv_kernels", [3, 5, 7])
        if not isinstance(kernels, (list, tuple)) or len(kernels) == 0:
            kernels = [3, 5, 7]
        self.conv_blocks_list = nn.ModuleList([
            DepthwiseSeparableConv1DBlock(
                d_model=self.d_model,
                kernel_size=int(kernels[i % len(kernels)]),
                dropout=self.dropout
            )
            for i in range(self.conv_blocks)
        ])

        # Global transformer branch
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.heads,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.transformer_layers)

        # Pooling module
        self.attn_pool = AttnPooling(self.d_model, self.dropout)

        # Learned CLS token (only used for CLS-based pooling modes)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Pooling output dimension
        if self.pooling in {"mean", "attn", "cls"}:
            self.pool_out_dim = self.d_model
        elif self.pooling == "minmaxcls":
            self.pool_out_dim = 3 * self.d_model
        else:
            raise RuntimeError("Unhandled pooling mode")

        # Sequence head: pool_out_dim -> d_model -> 1
        self.seq_head = nn.Sequential(
            nn.Linear(self.pool_out_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        # Optional token head (kept OFF by default to avoid breaking training loops)
        self.return_token_scores = _get_attr(args, "return_token_scores", False)
        if self.return_token_scores:
            self.token_head = nn.Linear(self.d_model, 1)

    def _rank_bucket_ids(self, ATP_R: torch.Tensor) -> torch.Tensor:
        """
        ATP_R: [B, N] (int-ish)
        returns bucket_id: [B, N] in [0..num_buckets-1]
        """
        r = ATP_R.long().clamp_min(0)
        bucket = torch.bucketize(r, self.rank_bucket_thresholds, right=True)
        return bucket.clamp_max(self.rank_bucket_emb.num_embeddings - 1)

    def _apply_group_feature_mask(self, feats: torch.Tensor) -> torch.Tensor:
        """
        During training optionally zero out feature groups:
        - top-k logp block
        - shape stats block (entropy/margin/loggap/cdf/p_tail + d_entropy/d2_entropy)
        - ATP block (log_raw_atp, normalized_ATP, d_atp, d2_atp)
        - rank block (rank, d_rank, rank_emb)
        """
        if (not self.training) or (not self.feature_masking) or (self.feature_mask_p <= 0):
            return feats

        B, N, Fdim = feats.shape
        device = feats.device

        # Indices based on concat order:
        i = 0
        sl_logp = slice(i, i + self.k); i += self.k
        sl_shape = slice(i, i + (1 + 1 + 1 + 1 + 4)); i += (1 + 1 + 1 + 1 + 4)  # entropy, margin, log_gap, p_tail, cdfs
        sl_atp = slice(i, i + 4); i += 4  # log_raw_atp, normalized_ATP, d_atp, d2_atp
        sl_rank = slice(i, i + 2); i += 2  # rank, d_rank
        sl_dent = slice(i, i + 2); i += 2  # d_entropy, d2_entropy
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
            sorted_TDS_normalized: [B, N, V]  (descending over V)
            normalized_ATP:        [B, N, 1]
            ATP_R:                [B, N]

        Output:
            logits: [B]
            (optionally token_logits: [B, seq_len, 1] if args.return_token_scores=True)
        """
        B, N, V = sorted_TDS_normalized.shape
        k = min(self.k, V)

        # 2.1 Top-k probabilities
        p_top = sorted_TDS_normalized[:, :, :k].to(torch.float32)          # [B,N,k]
        p_top = torch.clamp(p_top, min=0.0)
        p_tail = 1.0 - torch.sum(p_top, dim=-1, keepdim=True)              # [B,N,1]
        p_tail = torch.clamp(p_tail, min=0.0)

        logp_top = torch.log(p_top + self.eps)
        logp_top = torch.nan_to_num(
            logp_top,
            nan=torch.log(torch.tensor(self.eps, device=logp_top.device)),
            posinf=0.0,
            neginf=torch.log(torch.tensor(self.eps, device=logp_top.device))
        )

        # 2.2 Distribution-shape stats
        entropy_top = -torch.sum(p_top * logp_top, dim=-1, keepdim=True)   # [B,N,1]
        if k >= 2:
            margin = p_top[..., 0:1] - p_top[..., 1:2]                     # [B,N,1]
            log_gap = logp_top[..., 0:1] - logp_top[..., 1:2]              # [B,N,1]
            cdf2 = torch.sum(p_top[..., :2], dim=-1, keepdim=True)         # [B,N,1]
        else:
            margin = torch.zeros(B, N, 1, device=p_top.device, dtype=p_top.dtype)
            log_gap = torch.zeros_like(margin)
            cdf2 = torch.sum(p_top, dim=-1, keepdim=True)

        cdf5 = torch.sum(p_top[..., :min(5, k)], dim=-1, keepdim=True)
        cdf10 = torch.sum(p_top[..., :min(10, k)], dim=-1, keepdim=True)
        cdf20 = torch.sum(p_top, dim=-1, keepdim=True)                     # top-k mass

        # 2.3 Rank features + bucket embedding
        rank = ATP_R.unsqueeze(-1).to(torch.float32)                       # [B,N,1]
        bucket_ids = self._rank_bucket_ids(ATP_R)                          # [B,N]
        rank_emb = self.rank_bucket_emb(bucket_ids)                        # [B,N,E]

        # 2.4 Temporal deltas
        normalized_ATP = normalized_ATP.to(torch.float32)                  # [B,N,1]
        raw_ATP = normalized_ATP  # backward-compatible placeholder until dataset provides raw ATP
        log_raw_atp = torch.log(raw_ATP + self.eps)

        d_atp = _delta_leftpad(normalized_ATP)
        d2_atp = _delta_leftpad(d_atp)

        d_entropy = _delta_leftpad(entropy_top)
        d2_entropy = _delta_leftpad(d_entropy)

        d_rank = _delta_leftpad(rank)

        # 2.5 Final per-token feature vector
        # Pad logp_top to self.k if V < self.k
        if k < self.k:
            pad = torch.zeros(B, N, self.k - k, device=logp_top.device, dtype=logp_top.dtype)
            logp_top_full = torch.cat([logp_top, pad], dim=-1)             # [B,N,self.k]
        else:
            logp_top_full = logp_top                                       # [B,N,self.k]

        feats = torch.cat([
            logp_top_full,                 # [B,N,k]
            entropy_top,                   # [B,N,1]
            margin,                        # [B,N,1]
            log_gap,                       # [B,N,1]
            p_tail,                        # [B,N,1]
            cdf2, cdf5, cdf10, cdf20,      # [B,N,4]
            log_raw_atp,                   # [B,N,1]
            normalized_ATP,                # [B,N,1]
            d_atp, d2_atp,                 # [B,N,2]
            rank,                          # [B,N,1]
            d_rank,                        # [B,N,1]
            d_entropy, d2_entropy,         # [B,N,2]
            rank_emb                       # [B,N,E]
        ], dim=-1)                         # [B,N,F]

        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Group-wise normalization (preserve calibration scale) ---
        logp_block = feats[:, :, :self.k]
        rest_block = feats[:, :, self.k:]
        logp_block = self.logp_ln(logp_block)
        feats = torch.cat([logp_block, rest_block], dim=-1)

        feats = self._apply_group_feature_mask(feats)
        feats = self.feature_drop(feats)

        # 3.1 Token projection
        x = self.token_proj(feats)  # [B,N,D]

        # Prepend CLS token if needed (learned)
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)     # [B,1,D]
            x = torch.cat([cls, x], dim=1)                      # [B,1+N,D]

        # 3.2 Local conv blocks
        for blk in self.conv_blocks_list:
            x = blk(x)              # [B,seq_len,D]

        # 3.3 Global transformer
        # NOTE: no padding mask is wired yet (needs dataset to provide it).
        h = self.transformer(x)     # [B,seq_len,D]

        # 4.1 Pooling
        if self.pooling == "mean":
            # Mean over tokens (exclude CLS if present)
            if self.use_cls_token:
                pooled = h[:, 1:, :].mean(dim=1)
            else:
                pooled = h.mean(dim=1)

        elif self.pooling == "attn":
            # Attention pooling (exclude CLS if present)
            if self.use_cls_token:
                pooled = self.attn_pool(h[:, 1:, :])
            else:
                pooled = self.attn_pool(h)

        elif self.pooling == "cls":
            if not self.use_cls_token:
                raise RuntimeError("cls pooling requires CLS token")
            pooled = h[:, 0, :]  # [B,D]

        elif self.pooling == "minmaxcls":
            if not self.use_cls_token:
                raise RuntimeError("minmaxcls requires CLS token")
            cls_vec = h[:, 0, :]             # [B,D]
            tok = h[:, 1:, :]                # [B,N,D]
            tok_min = tok.min(dim=1).values  # [B,D]
            tok_max = tok.max(dim=1).values  # [B,D]
            pooled = torch.cat([cls_vec, tok_min, tok_max], dim=-1)  # [B,3D]

        else:
            raise RuntimeError(f"Unhandled pooling mode: {self.pooling}")

        # 4.2 Sequence head
        logits = self.seq_head(pooled)  # [B,1]

        if self.return_token_scores:
            token_logits = self.token_head(h)  # [B,seq_len,1]
            return logits.squeeze(-1), token_logits

        return logits.squeeze(-1)


# -----------------------------
# Update get_model mapping
# -----------------------------
def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    model_mapping = {
        # LOS-based
        'LOS++': LOS_PP_MultiScaleDeltaTransformer,
    }

    if args.probe_model in {'LOS-Net', 'LOS++', 'ATP_R_Transf'}:
        return model_mapping[args.probe_model](args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)
    elif args.probe_model in {'ATP_R_MLP'}:
        return model_mapping[args.probe_model](args=args, actual_sequence_length=actual_sequence_length)
    else:
        raise ValueError(f"Unknown model: {args.probe_model}")