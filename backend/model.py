import torch
from torch import nn

class TabTransformerModel(nn.Module):
    def __init__(self, num_features, cat_cardinalities, cat_embed_dim=32, d_model=64,
                 n_heads=4, n_layers=2, mlp_hidden=128, out_dim=1, dropout=0.15):
        super().__init__()
        self.num_features = num_features
        self.cat_cardinalities = cat_cardinalities
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, cat_embed_dim) for card in cat_cardinalities
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if len(cat_cardinalities) > 0:
            self.cat_proj = nn.Linear(len(cat_cardinalities) * cat_embed_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        if num_features > 0:
            self.num_proj = nn.Sequential(
                nn.Linear(num_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, out_dim)
        )
    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)
        tokens = []
        if len(self.cat_cardinalities) > 0 and x_cat.size(1) > 0:
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            cat_concat = torch.cat(cat_embeds, dim=1)
            cat_token = self.cat_proj(cat_concat).unsqueeze(1)
            tokens.append(cat_token)
        if self.num_features > 0:
            num_token = self.num_proj(x_num).unsqueeze(1)
            tokens.append(num_token)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens.insert(0, cls)
        x = torch.cat(tokens, dim=1)
        x = self.transformer(x)
        cls_output = x[:, 0]
        out = self.mlp(cls_output)
        return out.squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()