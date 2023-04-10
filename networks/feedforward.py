import torch
import torch.nn as nn
from typing import Optional

class FeedForward(nn.Module):
    """Simple Feed Forward Neural Network"""

    def __init__(self, d_input, d_output, d_hidden, n_hidden=0, dropout=0.6) -> None:
        super().__init__()
        d = d_input
        layers = []
        for n in range(n_hidden):
            layers.extend((
                nn.Linear(d, d_hidden),
                # nn.BatchNorm1d(d_hidden),
                nn.Dropout(dropout),
                nn.ReLU(True)))
            d = d_hidden
        layers.append(torch.nn.Linear(d, d_output))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
