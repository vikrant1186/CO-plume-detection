"""
Plume-detection CNN.

This is a RECONSTRUCTION of the architecture described in:

    Schuit, B. J., Maasakkers, J. D., Bijl, P., et al. (2023).
    Automated detection and monitoring of methane super-emitters using
    satellite data. Atmos. Chem. Phys. 23, 9071-9098.
    https://doi.org/10.5194/acp-23-9071-2023

The paper describes "two convolutional blocks followed by two fully connected
layers" operating on 32x32 single-channel scenes. Layer widths, kernel sizes,
dropout and the optimiser are NOT fully specified in the text, so the values
below are a reasonable reconstruction, not the published network.

>>> Say this explicitly wherever you present results. Do not claim to have
>>> reproduced their exact model. You are reproducing their *approach*.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PlumeCNN(nn.Module):
    """Binary plume / no-plume classifier on a single-channel 32x32 scene.

    The input is deliberately species-agnostic: the network sees a normalised
    scalar field, not a mixing ratio. That is what makes the CH4 -> CO
    transfer experiment meaningful -- the model can only be keying on spatial
    morphology, because the units have been normalised away.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # convolutional block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 32x32 -> 16x16
            # convolutional block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 16x16 -> 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (N,). Apply sigmoid for probabilities."""
        if x.ndim == 3:                            # (N, 32, 32) -> (N, 1, 32, 32)
            x = x.unsqueeze(1)
        return self.classifier(self.features(x)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = PlumeCNN()
    dummy = torch.randn(4, 32, 32)
    out = m(dummy)
    print(f"PlumeCNN: {count_parameters(m):,} trainable parameters")
    print(f"input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
