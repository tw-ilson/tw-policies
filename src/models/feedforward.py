import torch
import torch.nn as nn
from typing import Optional

class FeedForward(nn.Module):
    """Simple Feed Forward Neural Network"""

    def __init__(self, d_input, d_output, d_hidden, n_hidden=1, p_dropout=0.1) -> None:
        super().__init__()
        layers = [
            torch.nn.Linear(d_input, d_hidden),
            torch.nn.ReLU(True),
            nn.Dropout(p_dropout),
            ]
        for n in range(1, n_hidden):
            layers.extend((
                torch.nn.Linear(d_hidden, d_hidden),
                torch.nn.ReLU(True),
                nn.Dropout(p_dropout)))

        layers.append(torch.nn.Linear(d_hidden, d_output))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
