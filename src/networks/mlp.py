import torch
import torch.nn as nn

"""
This torch module is a simple Multilayer Perceptron generic neural netork with a single hidden layer followed by dropout and RELU activation
"""
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=32, lr=1e-3):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.Dropout(0.6), # To help normalize, prevent redundancy
                nn.ReLU(True), 
                nn.Linear(hidden_size, out_dim),
                )
    def forward(self, x):
        return self.layers(x)

