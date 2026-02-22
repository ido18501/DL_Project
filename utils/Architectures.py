"""
architectures.py — clean rewrite

This module preserves the ORIGINAL public API contract expected by the training code:

- get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape)
- model.forward(sorted_TDS_normalized, normalized_ATP, ATP_R) -> Tensor[B] with sigmoid probs

It also adds a new low-capacity, strongly-regularized time-series model (HALT-style GRU)
that is designed to generalize better and converge with fewer epochs.

Notes:
- Keeps existing model names ('LOS-Net', 'ATP_R_MLP', 'ATP_R_Transf') for compatibility.
- Adds new probe option: 'LOS_GRU'  (HALT-style).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from utils.constants import MODEL_VOCAB_SIZES


# ---------------------------
# Public factory (API contract)
# ---------------------------

def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    """
    Args:
        args: argparse-like namespace. Must contain at least:
              - probe_model
              - hidden_dim, dropout, num_layers, heads (for transformer models)
              - pool (optional; defaults handled)
              - rank_encoding in {'scale_encoding','one_hot_encoding'}
              - LLM string key for MODEL_VOCAB_SIZES
        max_sequence_length: int, N_max used for position embeddings / padding.
        actual_sequence_length: int, used by ATP_R_MLP (flattened).
        input_dim: int, last-dim of sorted_TDS_normalized (topk), typically 1000.
        input_shape: unused but kept for compatibility.
    """
    model_mapping = {
        # Baseline-compatible
        "LOS-Net": LOS_Net,
        "ATP_R_MLP": ATP_R_MLP,
        "ATP_R_Transf": ATP_R_Transf,

        # New: HALT-style time-series head
        "LOS_GRU": LOS_GRU,
    }

    if args.probe_model not in model_mapping:
        raise ValueError(f"Unknown model: {args.probe_model}")

    if args.probe_model in {"ATP_R_MLP"}:
        return model_mapping[args.probe_model](args=args, actual_sequence_length=actual_sequence_length)

    # Sequence models
    return model_mapping[args.probe_model](args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)


# ---------------------------
# Helpers
# ---------------------------

def _get_pool_type(args) -> str:
    pool = getattr(args, "pool", "cls")
    # keep backward compatibility and allow extra pool types used in your repo
    allowed = {"cls", "mean", "max", "mean_cls", "mean_max", "mean_max_cls"}
    if pool not in allowed:
        # fall back safely
        pool = "mean_max_cls"
    return pool


def _safe_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p: [B, N, V] probabilities (not necessarily summing to 1 over V, but usually do for top-k normalized)
    returns: [B, N, 1]
    """
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1, keepdim=True)


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x.clamp_min(eps)).log()


# ---------------------------
# Rank/ATP encodings (shared)
# ---------------------------

class RankATPEncoder(nn.Module):
    """
    Encodes (normalized_ATP, ATP_R) into a hidden vector per token.

    - normalized_ATP: [B, N, 1] float
    - ATP_R: [B, N] int (token rank/index)

    Output: [B, N, D]
    """
    def __init__(self, args, hidden_dim: int):
        super().__init__()
        self.args = args
        self.hidden_dim = hidden_dim

        # normalized_ATP modulation parameter
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, hidden_dim))

        rank_encoding = getattr(args, "rank_encoding", "scale_encoding")
        self.rank_encoding = rank_encoding

        if rank_encoding == "scale_encoding":
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, hidden_dim))
        elif rank_encoding == "one_hot_encoding":
            vocab_size = MODEL_VOCAB_SIZES[self.args.LLM]
            self.one_hot_embedding = nn.Embedding(vocab_size, hidden_dim)
        else:
            raise ValueError("Invalid rank_encoding. Choose 'scale_encoding' or 'one_hot_encoding'.")

    def compute_encoded_ATP_R(self, normalized_ATP: torch.Tensor, ATP_R: torch.Tensor) -> torch.Tensor:
        """
        Scale encoding from the original file:
        encoded_ATP_R = 2 * (0.5 - (ATP_R / vocab_size))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * param_for_ATP_R
        """
        vocab_size = MODEL_VOCAB_SIZES[self.args.LLM]
        encoded_ATP_R = 2.0 * (0.5 - (ATP_R.to(torch.float32) / float(vocab_size)))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R

    def forward(self, normalized_ATP: torch.Tensor, ATP_R: torch.Tensor) -> torch.Tensor:
        if self.rank_encoding == "scale_encoding":
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        else:
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)

        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        return encoded_ATP_R + encoded_normalized_ATP


# ---------------------------
# Models (baseline-compatible)
# ---------------------------

class ATP_R_MLP(nn.Module):
    """
    Baseline-compatible MLP using only (normalized_ATP, ATP_R), flattened over sequence.

    forward(sorted_TDS_normalized, normalized_ATP, ATP_R) -> [B]
    """
    def __init__(self, args, actual_sequence_length: int):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.actual_sequence_length = actual_sequence_length

        self.encoder = RankATPEncoder(args=args, hidden_dim=self.hidden_dim)

        self.lin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = self.hidden_dim * self.actual_sequence_length if i == 0 else self.hidden_dim
            out_dim = 1 if (i + 1) == self.num_layers else self.hidden_dim
            self.lin_layers.append(nn.Linear(in_dim, out_dim))
            if (i + 1) < self.num_layers:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        x = self.encoder(normalized_ATP=normalized_ATP, ATP_R=ATP_R)  # [B,N,D]
        x = x.flatten(start_dim=1)  # [B, N*D]

        for i, lin in enumerate(self.lin_layers):
            x = lin(x)
            if (i + 1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.sigmoid(x).squeeze(-1)


class ATP_R_Transf(nn.Module):
    """
    Baseline-compatible Transformer using only (normalized_ATP, ATP_R).

    forward(sorted_TDS_normalized, normalized_ATP, ATP_R) -> [B]
    """
    def __init__(self, args, max_sequence_length: int, input_dim: int = 1):
        super().__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = _get_pool_type(args)

        self.encoder = RankATPEncoder(args=args, hidden_dim=self.hidden_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(self.num_layers)
        ])

        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N+1, D]
        if self.pool == "cls":
            return x[:, 0]
        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "max":
            return x.max(dim=1).values
        if self.pool == "mean_cls":
            return 0.5 * (x.mean(dim=1) + x[:, 0])
        if self.pool == "mean_max":
            return 0.5 * (x.mean(dim=1) + x.max(dim=1).values)
        # mean_max_cls
        return (x.mean(dim=1) + x.max(dim=1).values + x[:, 0]) / 3.0

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        x = self.encoder(normalized_ATP=normalized_ATP, ATP_R=ATP_R).to(torch.float32)  # [B,N,D]
        b, n, _ = x.shape

        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls, x], dim=1)  # [B,N+1,D]

        pos_idx = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_idx)

        for layer in self.attention_layers:
            x = layer(x)

        x = self._pool(x)
        x = self.mlp_head(x)
        return self.sigmoid(x).squeeze(-1)


class LOS_Net(nn.Module):
    """
    Baseline-compatible LOS-Net:
    - Uses sorted_TDS_normalized projected to D/2
    - Concatenates with ATP features projected to D/2
    - Transformer over tokens (+ CLS)
    """
    def __init__(self, args, max_sequence_length: int, input_dim: int = 1):
        super().__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = _get_pool_type(args)

        assert self.hidden_dim % 2 == 0, "hidden_dim must be even for LOS_Net (split into two halves)."

        # ATP encoder produces D/2
        self.atp_encoder = RankATPEncoder(args=args, hidden_dim=self.hidden_dim // 2)

        # Project top-k probs/logits to D/2
        self.input_proj = nn.Linear(input_dim, self.hidden_dim // 2)

        # Tokens are concatenated to D
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(self.num_layers)
        ])

        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return x[:, 0]
        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "max":
            return x.max(dim=1).values
        if self.pool == "mean_cls":
            return 0.5 * (x.mean(dim=1) + x[:, 0])
        if self.pool == "mean_max":
            return 0.5 * (x.mean(dim=1) + x.max(dim=1).values)
        return (x.mean(dim=1) + x.max(dim=1).values + x[:, 0]) / 3.0  # mean_max_cls

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        # sorted_TDS_normalized: [B,N,V] (V ~ 1000), float
        tds = self.input_proj(sorted_TDS_normalized.to(torch.float32))  # [B,N,D/2]
        atp = self.atp_encoder(normalized_ATP=normalized_ATP, ATP_R=ATP_R).to(torch.float32)  # [B,N,D/2]
        x = torch.cat([tds, atp], dim=-1)  # [B,N,D]

        b, n, _ = x.shape
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls, x], dim=1)  # [B,N+1,D]

        pos_idx = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_idx)

        for layer in self.attention_layers:
            x = layer(x)

        x = self._pool(x)
        x = self.mlp_head(x)
        return self.sigmoid(x).squeeze(-1)


# ---------------------------
# New: LOS_GRU (HALT-style)
# ---------------------------

class LOS_GRU(nn.Module):
    """
    HALT-style time-series classifier for hallucination detection.

    Key idea:
    - Convert per-token top-k distribution into a small set of uncertainty/stat features.
    - Concatenate with ATP-based features (normalized_ATP + rank encoding).
    - Run a (bi)GRU over token time steps.
    - Pool + classify.

    This is designed to:
    - reduce overfitting (lower effective capacity, stronger inductive bias)
    - converge with fewer epochs
    """
    def __init__(self, args, max_sequence_length: int, input_dim: int = 1000):
        super().__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim

        # We use hidden_dim as GRU hidden size (keep it small in sweeps: 64/96/128)
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_layers = max(1, int(getattr(args, "num_layers", 1)))
        self.bidirectional = bool(getattr(args, "bidirectional", True))
        self.pool = _get_pool_type(args)

        # Feature settings
        self.use_entropy = True
        self.use_margin = True
        self.use_top_stats = True
        self.use_atp = True

        # ATP encoder -> small vector
        # We keep ATP encoding width modest to avoid overfitting.
        atp_dim = int(getattr(args, "atp_feature_dim", max(8, self.hidden_dim // 8)))
        self.atp_encoder = RankATPEncoder(args=args, hidden_dim=atp_dim) if self.use_atp else None

        # Build per-token feature dimension from top-k distribution
        # Features (all per token):
        # - entropy(topk)               [1]
        # - margin(p1 - p2)             [1]
        # - log(p1)                     [1]
        # - mean(p), std(p), max(p)     [3]
        # - mean(log p), std(log p)     [2]
        # Total (without ATP): 1 + 1 + 1 + 3 + 2 = 8
        base_feat_dim = 0
        if self.use_entropy: base_feat_dim += 1
        if self.use_margin: base_feat_dim += 2  # margin + log(p1)
        if self.use_top_stats: base_feat_dim += 5  # mean, std, max, mean_log, std_log

        total_feat_dim = base_feat_dim + (atp_dim if self.use_atp else 0)

        # Project token features to GRU input size
        gru_in_dim = int(getattr(args, "gru_input_dim", self.hidden_dim))
        self.feat_proj = nn.Sequential(
            nn.Linear(total_feat_dim, gru_in_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.gru = nn.GRU(
            input_size=gru_in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        # A small head (optionally with LayerNorm) to stabilize and reduce overfit
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def _token_features(self, sorted_TDS_normalized: torch.Tensor) -> torch.Tensor:
        """
        sorted_TDS_normalized: [B, N, V]
        returns: [B, N, base_feat_dim]
        """
        p = sorted_TDS_normalized.to(torch.float32)

        feats = []

        if self.use_entropy:
            feats.append(_safe_entropy(p))  # [B,N,1]

        if self.use_margin:
            # assume sorted descending => p[...,0] >= p[...,1]
            p1 = p[..., 0:1]
            p2 = p[..., 1:2] if p.size(-1) > 1 else torch.zeros_like(p1)
            feats.append(p1 - p2)           # margin [B,N,1]
            feats.append(_safe_log(p1))     # log(p1) [B,N,1]

        if self.use_top_stats:
            mean_p = p.mean(dim=-1, keepdim=True)
            std_p = p.std(dim=-1, keepdim=True, unbiased=False)
            max_p = p.max(dim=-1, keepdim=True).values

            logp = _safe_log(p)
            mean_logp = logp.mean(dim=-1, keepdim=True)
            std_logp = logp.std(dim=-1, keepdim=True, unbiased=False)

            feats.extend([mean_p, std_p, max_p, mean_logp, std_logp])

        return torch.cat(feats, dim=-1)

    def _pool_seq(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N, D]
        returns: [B, D] pooled
        """
        if self.pool == "mean":
            return h.mean(dim=1)
        if self.pool == "max":
            return h.max(dim=1).values
        if self.pool == "cls":
            # For GRU there is no CLS token; we interpret "cls" as last timestep.
            return h[:, -1]
        if self.pool == "mean_cls":
            return 0.5 * (h.mean(dim=1) + h[:, -1])
        if self.pool == "mean_max":
            return 0.5 * (h.mean(dim=1) + h.max(dim=1).values)
        # mean_max_cls
        return (h.mean(dim=1) + h.max(dim=1).values + h[:, -1]) / 3.0

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        # Base distribution-derived token features
        base = self._token_features(sorted_TDS_normalized)  # [B,N,F]

        # Optional ATP/rank features
        if self.use_atp:
            atp = self.atp_encoder(normalized_ATP=normalized_ATP, ATP_R=ATP_R).to(torch.float32)  # [B,N,atp_dim]
            feats = torch.cat([base, atp], dim=-1)
        else:
            feats = base

        x = self.feat_proj(feats)  # [B,N,gru_in_dim]

        # GRU
        h, _ = self.gru(x)  # [B,N,out_dim]

        pooled = self._pool_seq(h)  # [B,out_dim]
        logits = self.head(pooled)  # [B,1]

        return self.sigmoid(logits).squeeze(-1)

def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    """
    Args:
        args: argparse-like namespace. Must contain at least:
              - probe_model
              - hidden_dim, dropout, num_layers, heads (for transformer models)
              - pool (optional; defaults handled)
              - rank_encoding in {'scale_encoding','one_hot_encoding'}
              - LLM string key for MODEL_VOCAB_SIZES
        max_sequence_length: int, N_max used for position embeddings / padding.
        actual_sequence_length: int, used by ATP_R_MLP (flattened).
        input_dim: int, last-dim of sorted_TDS_normalized (topk), typically 1000.
        input_shape: unused but kept for compatibility.
    """
    model_mapping = {
        # Baseline-compatible
        "LOS-Net": LOS_Net,
        "ATP_R_MLP": ATP_R_MLP,
        "ATP_R_Transf": ATP_R_Transf,

        # New: HALT-style time-series head
        "LOS_GRU": LOS_GRU,
    }

    if args.probe_model not in model_mapping:
        raise ValueError(f"Unknown model: {args.probe_model}")

    if args.probe_model in {"ATP_R_MLP"}:
        return model_mapping[args.probe_model](args=args, actual_sequence_length=actual_sequence_length)

    # Sequence models
    return model_mapping[args.probe_model](args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)


# ---------------------------
# Helpers
# ---------------------------

def _get_pool_type(args) -> str:
    pool = getattr(args, "pool", "cls")
    # keep backward compatibility and allow extra pool types used in your repo
    allowed = {"cls", "mean", "max", "mean_cls", "mean_max", "mean_max_cls"}
    if pool not in allowed:
        # fall back safely
        pool = "mean_max_cls"
    return pool


def _safe_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p: [B, N, V] probabilities (not necessarily summing to 1 over V, but usually do for top-k normalized)
    returns: [B, N, 1]
    """
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1, keepdim=True)


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x.clamp_min(eps)).log()


# ---------------------------
# Rank/ATP encodings (shared)
# ---------------------------

class RankATPEncoder(nn.Module):
    """
    Encodes (normalized_ATP, ATP_R) into a hidden vector per token.

    - normalized_ATP: [B, N, 1] float
    - ATP_R: [B, N] int (token rank/index)

    Output: [B, N, D]
    """
    def __init__(self, args, hidden_dim: int):
        super().__init__()
        self.args = args
        self.hidden_dim = hidden_dim

        # normalized_ATP modulation parameter
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, hidden_dim))

        rank_encoding = getattr(args, "rank_encoding", "scale_encoding")
        self.rank_encoding = rank_encoding

        if rank_encoding == "scale_encoding":
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, hidden_dim))
        elif rank_encoding == "one_hot_encoding":
            vocab_size = MODEL_VOCAB_SIZES[self.args.LLM]
            self.one_hot_embedding = nn.Embedding(vocab_size, hidden_dim)
        else:
            raise ValueError("Invalid rank_encoding. Choose 'scale_encoding' or 'one_hot_encoding'.")

    def compute_encoded_ATP_R(self, normalized_ATP: torch.Tensor, ATP_R: torch.Tensor) -> torch.Tensor:
        """
        Scale encoding from the original file:
        encoded_ATP_R = 2 * (0.5 - (ATP_R / vocab_size))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * param_for_ATP_R
        """
        vocab_size = MODEL_VOCAB_SIZES[self.args.LLM]
        encoded_ATP_R = 2.0 * (0.5 - (ATP_R.to(torch.float32) / float(vocab_size)))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R

    def forward(self, normalized_ATP: torch.Tensor, ATP_R: torch.Tensor) -> torch.Tensor:
        if self.rank_encoding == "scale_encoding":
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        else:
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)

        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        return encoded_ATP_R + encoded_normalized_ATP


# ---------------------------
# Models (baseline-compatible)
# ---------------------------

class ATP_R_MLP(nn.Module):
    """
    Baseline-compatible MLP using only (normalized_ATP, ATP_R), flattened over sequence.

    forward(sorted_TDS_normalized, normalized_ATP, ATP_R) -> [B]
    """
    def __init__(self, args, actual_sequence_length: int):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.actual_sequence_length = actual_sequence_length

        self.encoder = RankATPEncoder(args=args, hidden_dim=self.hidden_dim)

        self.lin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = self.hidden_dim * self.actual_sequence_length if i == 0 else self.hidden_dim
            out_dim = 1 if (i + 1) == self.num_layers else self.hidden_dim
            self.lin_layers.append(nn.Linear(in_dim, out_dim))
            if (i + 1) < self.num_layers:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        x = self.encoder(normalized_ATP=normalized_ATP, ATP_R=ATP_R)  # [B,N,D]
        x = x.flatten(start_dim=1)  # [B, N*D]

        for i, lin in enumerate(self.lin_layers):
            x = lin(x)
            if (i + 1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.sigmoid(x).squeeze(-1)


class ATP_R_Transf(nn.Module):
    """
    Baseline-compatible Transformer using only (normalized_ATP, ATP_R).

    forward(sorted_TDS_normalized, normalized_ATP, ATP_R) -> [B]
    """
    def __init__(self, args, max_sequence_length: int, input_dim: int = 1):
        super().__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = _get_pool_type(args)

        self.encoder = RankATPEncoder(args=args, hidden_dim=self.hidden_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(self.num_layers)
        ])

        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N+1, D]
        if self.pool == "cls":
            return x[:, 0]
        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "max":
            return x.max(dim=1).values
        if self.pool == "mean_cls":
            return 0.5 * (x.mean(dim=1) + x[:, 0])
        if self.pool == "mean_max":
            return 0.5 * (x.mean(dim=1) + x.max(dim=1).values)
        # mean_max_cls
        return (x.mean(dim=1) + x.max(dim=1).values + x[:, 0]) / 3.0

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        x = self.encoder(normalized_ATP=normalized_ATP, ATP_R=ATP_R).to(torch.float32)  # [B,N,D]
        b, n, _ = x.shape

        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls, x], dim=1)  # [B,N+1,D]

        pos_idx = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_idx)

        for layer in self.attention_layers:
            x = layer(x)

        x = self._pool(x)
        x = self.mlp_head(x)
        return self.sigmoid(x).squeeze(-1)


class LOS_Net(nn.Module):
    """
    Baseline-compatible LOS-Net:
    - Uses sorted_TDS_normalized projected to D/2
    - Concatenates with ATP features projected to D/2
    - Transformer over tokens (+ CLS)
    """
    def __init__(self, args, max_sequence_length: int, input_dim: int = 1):
        super().__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = _get_pool_type(args)

        assert self.hidden_dim % 2 == 0, "hidden_dim must be even for LOS_Net (split into two halves)."

        # ATP encoder produces D/2
        self.atp_encoder = RankATPEncoder(args=args, hidden_dim=self.hidden_dim // 2)

        # Project top-k probs/logits to D/2
        self.input_proj = nn.Linear(input_dim, self.hidden_dim // 2)

        # Tokens are concatenated to D
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(self.num_layers)
        ])

        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return x[:, 0]
        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "max":
            return x.max(dim=1).values
        if self.pool == "mean_cls":
            return 0.5 * (x.mean(dim=1) + x[:, 0])
        if self.pool == "mean_max":
            return 0.5 * (x.mean(dim=1) + x.max(dim=1).values)
        return (x.mean(dim=1) + x.max(dim=1).values + x[:, 0]) / 3.0  # mean_max_cls

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        # sorted_TDS_normalized: [B,N,V] (V ~ 1000), float
        tds = self.input_proj(sorted_TDS_normalized.to(torch.float32))  # [B,N,D/2]
        atp = self.atp_encoder(normalized_ATP=normalized_ATP, ATP_R=ATP_R).to(torch.float32)  # [B,N,D/2]
        x = torch.cat([tds, atp], dim=-1)  # [B,N,D]

        b, n, _ = x.shape
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls, x], dim=1)  # [B,N+1,D]

        pos_idx = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_idx)

        for layer in self.attention_layers:
            x = layer(x)

        x = self._pool(x)
        x = self.mlp_head(x)
        return self.sigmoid(x).squeeze(-1)


# ---------------------------
# New: LOS_GRU (HALT-style)
# ---------------------------

class LOS_GRU(nn.Module):
    def __init__(self, args, max_sequence_length: int, input_dim: int = 1000):
        super().__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim

        # GRU capacity
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_layers = max(1, int(getattr(args, "num_layers", 1)))
        self.bidirectional = bool(getattr(args, "bidirectional", True))
        self.pool = _get_pool_type(args)

        # Feature switches (must match _token_features)
        self.use_entropy = True
        self.use_margin = True
        self.use_top_stats = True
        self.use_atp = True

        # If you added dynamics features, keep this True
        self.use_dynamics = True
        self.roll_window = int(getattr(args, "roll_window", 4))

        # ATP encoder (keep small)
        self.atp_dim = int(getattr(args, "atp_feature_dim", max(8, self.hidden_dim // 8)))
        self.atp_encoder = RankATPEncoder(args=args, hidden_dim=self.atp_dim) if self.use_atp else None

        # Compute feature dims EXACTLY (don’t hand count elsewhere)
        base_feat_dim = self._base_feat_dim()
        total_feat_dim = base_feat_dim + (self.atp_dim if self.use_atp else 0)

        # Project token features to GRU input size
        self.gru_in_dim = int(getattr(args, "gru_input_dim", self.hidden_dim))
        self.feat_proj = nn.Sequential(
            nn.Linear(total_feat_dim, self.gru_in_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.gru = nn.GRU(
            input_size=self.gru_in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        self.out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

        # Optional one-time debug to ensure dims match at runtime
        self._debug_feat_dim_once = bool(getattr(args, "debug_feat_dim", False))

    def _base_feat_dim(self) -> int:
        """
        MUST match exactly what _token_features() appends.
        """
        d = 0

        if self.use_entropy:
            d += 1  # H
            if self.use_dynamics:
                d += 1  # dH
                d += 1  # roll_mean(dH)
                d += 1  # roll_var(dH)

        if self.use_margin:
            d += 1  # margin
            d += 1  # log(p1)
            if self.use_dynamics:
                d += 1  # dlogp1
                d += 1  # dmargin

        if self.use_top_stats:
            d += 5  # mean_p, std_p, max_p, mean_logp, std_logp

        return d
    def _token_features(self, sorted_TDS_normalized: torch.Tensor) -> torch.Tensor:
        """
        sorted_TDS_normalized: [B, N, V]
        returns: [B, N, base_feat_dim]
        """
        p = sorted_TDS_normalized.to(torch.float32)

        feats = []

        # dynamics (very HALT/EPR-aligned)
        if self.use_entropy:
            H = _safe_entropy(p)  # [B,N,1]
            dH = _delta_feat(H)
            feats.append(dH)
            feats.append(_rolling_mean(dH, w=4))
            feats.append(_rolling_var(dH, w=4))

        if self.use_margin:
            p1 = p[..., 0:1]
            p2 = p[..., 1:2] if p.size(-1) > 1 else torch.zeros_like(p1)
            margin = (p1 - p2)
            logp1 = _safe_log(p1)

            dlogp1 = _delta_feat(logp1)
            dmargin = _delta_feat(margin)

            feats.append(dlogp1)
            feats.append(dmargin)

        if self.use_top_stats:
            mean_p = p.mean(dim=-1, keepdim=True)
            std_p = p.std(dim=-1, keepdim=True, unbiased=False)
            max_p = p.max(dim=-1, keepdim=True).values

            logp = _safe_log(p)
            mean_logp = logp.mean(dim=-1, keepdim=True)
            std_logp = logp.std(dim=-1, keepdim=True, unbiased=False)

            feats.extend([mean_p, std_p, max_p, mean_logp, std_logp])

        return torch.cat(feats, dim=-1)

    def _pool_seq(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N, D]
        returns: [B, D] pooled
        """
        if self.pool == "mean":
            return h.mean(dim=1)
        if self.pool == "max":
            return h.max(dim=1).values
        if self.pool == "cls":
            # For GRU there is no CLS token; we interpret "cls" as last timestep.
            return h[:, -1]
        if self.pool == "mean_cls":
            return 0.5 * (h.mean(dim=1) + h[:, -1])
        if self.pool == "mean_max":
            return 0.5 * (h.mean(dim=1) + h.max(dim=1).values)
        # mean_max_cls
        return (h.mean(dim=1) + h.max(dim=1).values + h[:, -1]) / 3.0

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        # Base distribution-derived token features
        base = self._token_features(sorted_TDS_normalized)  # [B,N,F]

        # Optional ATP/rank features
        if self.use_atp:
            atp = self.atp_encoder(normalized_ATP=normalized_ATP, ATP_R=ATP_R).to(torch.float32)  # [B,N,atp_dim]
            feats = torch.cat([base, atp], dim=-1)
        else:
            feats = base

        x = self.feat_proj(feats)  # [B,N,gru_in_dim]

        # GRU
        h, _ = self.gru(x)  # [B,N,out_dim]

        pooled = self._pool_seq(h)  # [B,out_dim]
        logits = self.head(pooled)  # [B,1]

        return self.sigmoid(logits).squeeze(-1)