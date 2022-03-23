from typing import List

import torch
import numpy as np
import torch.nn as nn

from gym.spaces import Space, Discrete


class PolicyMLP(nn.Module):
    ''' MLP Network layers for second half of Actor Network'''

    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.sequence = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(True),
                nn.Linear(128, 128),
                nn.ReLU(True),
                nn.Linear(128, 128),
                nn.ReLU(True),
                nn.Linear(128, 128),
                nn.ReLU(True),
                nn.Linear(128, output_size),
            )

    def forward(self, x):
        return self.sequence(x)


class CNN(nn.Module):
    ''' Convolutional half module of Actor network to predict actions from state '''

    def __init__(self):
        '''
        Input:
            84x84 grayscale image
        '''
        super().__init__()

        self.img_shape = img_shape

        self.sequence = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=2),
                nn.ReLU(True),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, kernel_size=4, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, kernel_size=4, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, kernel_size=4, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
            )

        self.output_size = self.compute_output_size(self.img_shape)

    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D grayscale float vector """
        def rgb2gray(I):
            return np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])

        I = I[35:195] # crop
        I = rgb2gray(I.astype(np.float32))
        I = I[::2,::2,0] # downsample by factor of 2

        return I.ravel()

    def forward(self, x):
        return self.sequence(x)

    def compute_output_size(self, img_shape) -> int:
        x = torch.zeros( img_shape, dtype=torch.float32)

        # add batch dimension
        x = x.unsqueeze(0)

        out = torch.flatten(self.forward(x))
        return out.shape[1]

class PolicyNetwork(nn.Module):
    '''The full policy network composed of convolutional and MLP parts '''

    def __init__(self,
            from_image:bool=False,
            state_space:Space=None,
            action_space:Space=None):

        super().__init__()


        self.state_space = state_space
        self.action_space = action_space
        if not from_image:
            assert len(state_space.shape) == 1
            self.layers = PolicyMLP( \
                    state_space.shape, \
                    action_space.shape)
        else:
            self.layers = nn.Sequential(CNN()
        
        self.output_layer = nn.LogSoftmax()

    def forward(self, state):
        return self.output_layer(self.layers(state))
