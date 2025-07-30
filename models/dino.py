import timm
import torch.nn as nn


class RadioDINO(nn.Module):
    def __init__(self):
        super(RadioDINO, self).__init__()
        self.encoder = timm.create_model("hf_hub:Snarcy/RadioDino-s16", pretrained=True)
        self.encoder.head = nn.Identity()
    
    def forward(self, x):
        return self.encoder(x)
