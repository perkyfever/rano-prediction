import sys
from pathlib import Path

BMMAE_PATH = Path(__file__).parent.parent / "BM-MAE"
sys.path.append(BMMAE_PATH.as_posix())

import torch
import torch.nn as nn

from bmmae.model import ViTEncoder
from bmmae.tokenizers import MRITokenizer


class BMMAE(nn.Module):
    def __init__(
        self,
        encoder_dropout: float,
        dropout: float,
        num_classes: int = 4,
        logits: bool = False,
        image_keys: list[str] = ["T1", "T1CE", "T2", "FLAIR"],
        unfreeze_layers: list[str] = ["blocks.11"]
    ):
        super(BMMAE, self).__init__()
        self.cls_token_idx = 0
        
        # SETTING UP ENCODER
        tokenizers = {
            modality: MRITokenizer(
                patch_size=(16, 16, 16),
                img_size=(128, 128, 128),
                hidden_size=768,
            )
            for modality in image_keys
        }
        self.encoder = ViTEncoder(
            modalities=image_keys,
            tokenizers=tokenizers,
            cls_token=True,
            dropout_rate=encoder_dropout
        )

        # LOADING WEIGHTS
        state_dict = torch.load(BMMAE_PATH / "pretrained_models" / "bmmae.pth")
        self.encoder.load_state_dict(state_dict, strict=False)

        for name, param in self.encoder.named_parameters():
            if all(x not in name for x in unfreeze_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self.delta = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
            nn.LayerNorm(self.encoder.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        clf_layers = [
            nn.Linear(3 * self.encoder.hidden_size, self.encoder.hidden_size),
            nn.LayerNorm(self.encoder.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.hidden_size, num_classes),
        ]
        
        if not logits:
            clf_layers.append(nn.Softmax(dim=-1))

        self.clf_head = nn.Sequential(*clf_layers)

    def forward(self, baseline, followup):
        baseline_embed = self.encoder(baseline)[:, self.cls_token_idx]
        followup_embed = self.encoder(followup)[:, self.cls_token_idx]
        
        delta_embed = self.delta(followup_embed - baseline_embed)
        joint_embed = torch.cat([baseline_embed, followup_embed, delta_embed], dim=1)
        
        outputs = self.clf_head(joint_embed)

        return outputs
