"""Small classification heads used for representation probes."""

from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.layers = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.layers(features)


class MLPClassifier(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.layers(features)
