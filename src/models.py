from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class EffNet(nn.Module):
    def __init__(self, arch_name: str, num_classes: int) -> None:
        super().__init__()
        self.model = EfficientNet.from_pretrained(arch_name)
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out
