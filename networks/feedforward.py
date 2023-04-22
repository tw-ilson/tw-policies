import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """Simple Feed Forward Neural Network"""

    def __init__(self, d_input, d_output, d_hidden, n_hidden=0, dropout=0.6, batchnorm=False) -> None:
        super().__init__()
        d = d_input
        layers = []
        for n in range(n_hidden):
            layers.append(nn.Linear(d, d_hidden))
            if batchnorm and n == 1:
                layers.append(nn.BatchNorm1d(d_hidden))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(True))
            d = d_hidden
        layers.append(torch.nn.Linear(d, d_output))
        self.layers = torch.nn.Sequential(*layers)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(init_weights)


    def forward(self, x):
        return self.layers(x)
