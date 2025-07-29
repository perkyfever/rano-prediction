import torch
import torch.nn as nn

from typing import Callable


class BaselineModel(nn.Module):
    def __init__(
        self,
        encoder: Callable,
        emb_dim: int,
        dropout: float,
        num_classes: int = 4,
        logits: bool = False,
        image_keys: list[str] = ["T1", "T1CE", "T2", "FLAIR"],
    ):
        super(BaselineModel, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.image_keys = image_keys
        self.image_encoders = nn.ModuleList([encoder() for _ in range(len(self.image_keys))])

        clf_layers = [
            nn.Linear(len(self.image_keys) * self.emb_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.LayerNorm(self.emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.emb_dim // 2, self.num_classes),
        ]

        if not logits:
            clf_layers.append(nn.Softmax(dim=-1))

        self.clf_head = nn.Sequential(*clf_layers)

    def forward(self, baseline, followup):
        baseline_embeds = []
        followup_embeds = []

        for image_key, image_encoder in zip(self.image_keys, self.image_encoders):
            baseline_embed = image_encoder(baseline[image_key])
            followup_embed = image_encoder(followup[image_key])

            baseline_embeds.append(baseline_embed)
            followup_embeds.append(followup_embed)

        baseline_fused_embed = torch.cat(baseline_embeds, dim=1)
        followup_fused_embed = torch.cat(followup_embeds, dim=1)

        delta_embed = followup_fused_embed - baseline_fused_embed
        outputs = self.clf_head(delta_embed)

        return outputs
