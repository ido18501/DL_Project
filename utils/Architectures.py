import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.constants import MODEL_VOCAB_SIZES
from einops import repeat
from vit_pytorch import ViT
from utils.Architectures_utils import *
def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    model_mapping = {
        # LOS-based
        'LOS-Net': LOS_Net,
        'ATP_R_MLP': ATP_R_MLP,
        'ATP_R_Transf': ATP_R_Transf,
    }
    
    if args.probe_model in {'LOS-Net', 'ATP_R_Transf'}:
        return model_mapping[args.probe_model](args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)
    elif args.probe_model in {'ATP_R_MLP'}:
        return model_mapping[args.probe_model](args=args, actual_sequence_length=actual_sequence_length)
    else:
        raise ValueError(f"Unknown model: {args.probe_model}")
    

######################## LOS ########################
class ATP_R_MLP(nn.Module):

    def __init__(self, args, actual_sequence_length):

        super(ATP_R_MLP, self).__init__()        
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.actual_sequence_length = actual_sequence_length
        
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim,
            # sparse=True
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")

        
        # Linear layers
        self.lin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim if i > 0 else self.hidden_dim * self.actual_sequence_length
            out_dim = self.hidden_dim if (i+1) < self.num_layers else 1
            self.lin_layers.append(nn.Linear(in_dim, out_dim))
            if (i+1) < self.num_layers:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Output act
        self.sigmoid = nn.Sigmoid()
    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):


        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
                    
        # Encoding normalized mark
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        x = encoded_ATP_R + encoded_normalized_ATP
        x = x.flatten(start_dim=1)
        
        for i in range(self.num_layers):
            x = self.lin_layers[i](x)
            if (i+1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)

        # Ido and yaniv- trying logits instead of sigmoid
        return x.squeeze(-1)


class ATP_R_Transf(nn.Module):
    
    def __init__(self, args, max_sequence_length, input_dim=1):
        
        super(ATP_R_Transf, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = args.hidden_dim
        # Ido and Yaniv- add attn pooling layer
        self.attn_pool = nn.Linear(self.hidden_dim, 1)

        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = args.pool
        assert self.pool in {'cls', 'mean', 'mean_max','mean_cls', 'mean_max_cls', 'attn'},('pool type must be either '
                                                                                            'cls '
                                                                                     '(cls token),mean (mean pooling)' 
                                                                                     ' or mean_max (mean-max polling)'
                                                                                     'mean/mean_max _cls is '
                                                                                     'also allowed to combine; attn is'
                                                                                            ' supported for attention'
                                                                                            ' layer')

        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim,
            # sparse=True
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
        
        

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Positional embeddings with a predefined max sequence length
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        # Transformer encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])

        # Classification head

        # Ido and Yaniv:
        # first change done here - head's input dim modification in order to support mean-max polling
        head_in = self.hidden_dim
        if self.pool == 'mean_max':
            head_in *= 2
        self.mlp_head = nn.Linear(head_in, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
            
        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
                    
        # Encoding normalized mark
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        x = encoded_ATP_R + encoded_normalized_ATP

    
        # Add [CLS] token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # Shape: [B, 1, hidden_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, N+1, hidden_dim]

        # Generate positional indices and add embeddings
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)  # Shape: [1, N+1]
        pos_embeddings = self.pos_embedding(pos_indices)  # Shape: [1, N+1, hidden_dim]
        x += pos_embeddings

        # Pass through Transformer layers
        for layer in self.attention_layers:
            x = layer(x)  # Shape remains [B, N+1, hidden_dim]

        # Pooling: Use the CLS token
        # Ido and Yaniv: another change here - we don't want to average over cls, and we want to support mean-max
        # second change: allowing mean/mean_max to combine cls
        # third change: adding support for attention pooling
        x_tokens = x[:, 1:, :]
        x_cls = x[:, 0, :]
        # Ido and Yaniv- try to mask
        token_mask = (sorted_TDS_normalized.abs().sum(dim=-1) > 0)

        # masked mean
        mask_f = token_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp(min=1e-6)
        x_mean = (x_tokens * mask_f).sum(dim=1) / denom

        # masked max
        x_masked = x_tokens.masked_fill(~token_mask.unsqueeze(-1), float('-inf'))
        x_max = x_masked.max(dim=1).values  # [B, d]
        x = None
        if self.pool == 'cls':
            x = x_cls
        elif self.pool == 'mean':
            x = x_mean
        elif self.pool == 'mean_max':
            x = torch.cat([x_mean, x_max], dim=-1)
        elif self.pool == 'mean_cls':
            x = torch.cat([x_mean, x_cls], dim=-1)
        elif self.pool == 'mean_max_cls':
            x = torch.cat([x_mean, x_max, x_cls], dim=-1)
        elif self.pool == 'attn':
            scores = self.attn_pool(x_tokens).squeeze(-1)
            scores = scores.masked_fill(~token_mask, float('-inf'))
            weights = torch.softmax(scores, dim=1)

            # Ido and Yaniv - store attention entropy for regularization (robustness)
            eps = 1e-12
            self.attn_entropy = -(weights * (weights + eps).log()).sum(dim=1).mean()

            x = (x_tokens * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError("Pooling type is not supported")

        # Final classification head
        x = self.mlp_head(x)  # Shape: [B, 1]

        # Ido and yaniv- trying logits instead of sigmoid
        return x.squeeze(-1)
    

class LOS_Net(nn.Module):
    def __init__(self, args, max_sequence_length, input_dim=1):
        super().__init__()
        
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        # Ido and Yaniv- add attn pooling layer
        self.attn_pool = nn.Linear(self.hidden_dim, 1)
        # Ido and Yaniv - project augmented token features back to model hidden_dim
        self.fuse_proj = nn.Linear(self.hidden_dim + self.hidden_dim // 2, self.hidden_dim)

        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = args.pool
        assert self.pool in {'cls', 'mean', 'mean_max', 'mean_cls', 'mean_max_cls', 'attn'}, (
            'pool type must be either '
            'cls '
            '(cls token),mean (mean pooling)'
            ' or mean_max (mean-max polling)'
            'mean/mean_max _cls is '
            'also allowed to combine; attn is'
            ' supported for attention'
            ' layer')
        
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))


        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))
        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim // 2,
            # sparse=True
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
        
        
        
        # Input embedding layer
        self.input_proj = nn.Linear(input_dim, self.hidden_dim // 2)

        # Ido and Yaniv - additional uncertainty features (entropy/margin/top1) projection
        self.uncertainty_proj = nn.Linear(4, self.hidden_dim // 2)


        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)
        
        # Transformer encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Classification head
        # Ido and Yaniv:
        # first change done here - head's input dim modification in order to support mean-max polling
        # second change - combining cls with man and mean-max
        head_in = self.hidden_dim
        if self.pool in {'mean_max', 'mean_cls'}:
            head_in *= 2
        elif self.pool == 'mean_max_cls':
            head_in = 3 * self.hidden_dim
        self.mlp_head = nn.Linear(head_in, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        """
        Forward pass for LOS_Net.

        Args:
            sorted_TDS_normalized (torch.Tensor): Shape [B, N, V].
            normalized_ATP (torch.Tensor): Shape [B, N, 1].
            ATP_R (torch.Tensor): Shape [B, N].
            sigmoid (bool): Whether to apply sigmoid activation. Default is True.

        Returns:
            torch.Tensor: Output tensor of shape [B, 1] (if sigmoid=True) or raw logits (if sigmoid=False).
        """
        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
            
        # Ido and Yaniv- token masking
        token_mask = (sorted_TDS_normalized.abs().sum(dim=-1) > 0)
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP

        #if self.training and hasattr(self, "keep_mask"):
         #   token_mask = token_mask & self.keep_mask
        
        
        # Encoding normalized vocab
        encoded_sorted_TDS_normalized = self.input_proj(sorted_TDS_normalized.to(torch.float32))

        # Ido and Yaniv - compute uncertainty features from top-K probabilities (robust)
        p = sorted_TDS_normalized.to(torch.float32)

        if self.training:
            sigma = getattr(self.args, "feat_noise", 0.0)  # default 0.0 is safer
            if sigma > 0:
                p = p + sigma * torch.randn_like(p)

        p = torch.clamp(p, 0.0, 1.0)

        # Ido and Yaniv - we use the same p for the main embedding, thus overriding
        encoded_sorted_TDS_normalized = self.input_proj(p)

        # Ido and Yaniv- uncertainty embeddings including gini (KL divergance)
        eps = 1e-12
        p1 = p[..., 0]
        margin = p[..., 0] - p[..., 1] if p.size(-1) >= 2 else torch.zeros_like(p1)
        entropy = -(p * (p.clamp_min(eps)).log()).sum(dim=-1)  # [B, N]
        gini = (p * p).sum(dim=-1)

        u = torch.stack([p1, margin, entropy, gini], dim=-1)
        encoded_uncertainty = self.uncertainty_proj(u)

        # Concatenating embeddings
        x = torch.cat((encoded_sorted_TDS_normalized, encoded_ATP_R + encoded_normalized_ATP), dim=-1)

        # Ido and Yaniv - include uncertainty embedding in the token representation
        x = torch.cat(
            (encoded_sorted_TDS_normalized, encoded_uncertainty, encoded_ATP_R + encoded_normalized_ATP),
            dim=-1
        )

        # Ido and Yaniv - fuse to hidden_dim for transformer
        x = self.fuse_proj(x)

        # Ido and Yaniv - token dropout
        if self.training:
            drop_p = getattr(self.args, "token_dropout", 0.2)  # try 0.1–0.3
            if drop_p > 0:
                keep = (torch.rand_like(token_mask.float()) > drop_p) & token_mask
                x = x * keep.unsqueeze(-1).float()
        # Adding CLS token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Positional embeddings
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x += self.pos_embedding(pos_indices)
        
        # Transformer layers
        for layer in self.attention_layers:
            x = layer(x)
        
        # Pooling
        # Ido and Yaniv: another change here - we don't want to average over cls, and we want to support mean-max
        # second change: allowing mean/mean_max to combine cls
        # third change: adding support for attention pooling
        x_tokens = x[:, 1:, :]
        x_cls = x[:, 0, :]
        # Ido and Yaniv- try to mask
        token_mask = (sorted_TDS_normalized.abs().sum(dim=-1) > 0)

        # masked mean
        mask_f = token_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp(min=1e-6)
        x_mean = (x_tokens * mask_f).sum(dim=1) / denom

        # masked max
        x_masked = x_tokens.masked_fill(~token_mask.unsqueeze(-1), float('-inf'))
        x_max = x_masked.max(dim=1).values  # [B, d]
        x = None
        if self.pool == 'cls':
            x = x_cls
        elif self.pool == 'mean':
            x = x_mean
        elif self.pool == 'mean_max':
            x = torch.cat([x_mean, x_max], dim=-1)
        elif self.pool == 'mean_cls':
            x = torch.cat([x_mean, x_cls], dim=-1)
        elif self.pool == 'mean_max_cls':
            x = torch.cat([x_mean, x_max, x_cls], dim=-1)
        elif self.pool == 'attn':
            scores = self.attn_pool(x_tokens).squeeze(-1)
            scores = scores.masked_fill(~token_mask, float('-inf'))
            weights = torch.softmax(scores, dim=1)

            # Ido and Yaniv - store attention entropy for regularization (robustness)
            eps = 1e-12
            self.attn_entropy = -(weights * (weights + eps).log()).sum(dim=1).mean()

            x = (x_tokens * weights.unsqueeze(-1)).sum(dim=1)

        else:
            raise ValueError("Pooling type is not supported")

        # Classification head
        x = self.mlp_head(x)

        # Ido and yaniv- trying logits instead of sigmoid
        return x.squeeze(-1)

   
######################## LOS ########################
