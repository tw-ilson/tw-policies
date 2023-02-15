import torch
from torch import nn
import numpy as np
from models.cnn import CNN
from models.reward import AbstractReturns

class ValueNetworkReturns(AbstractReturns, torch.nn.Module):
    def __init__(self, input_shape, alpha=1e-3, from_image=False) -> None:
        """Produces value estimation for state-action pairs 

        Args:
            cnn: the feature extractor to apply to the image
            input_shape (tuple, optional): Shape of the image (C, H, W). Should be square.
            NOTES: have 2 predictions of direct joint values: (base and gripper), two predictions
            of combinations of motors: (forward z values, and vertical position adjust)
        """
        super().__init__()
        # (B, C, H, W)

        self.alpha = alpha

        mlp = []
        #Constructed with convolutional neural network feature extractor provided.
        self.input_shape = input_shape
        if from_image:
            self.cnn = CNN(input_shape)
            mlp.append(
                torch.nn.Linear(self.cnn.out_size, 256))
        else:
            mlp.append(torch.nn.Linear(self.input_shape, 256))
            


        mlp.extend((
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            # predict values of state
            torch.nn.Linear(256, 1))
        )

        self.mlp = nn.Sequential(*mlp)

        self.mse = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=self.alpha)

    def __call__(self, state):
        return self.forward(state).item()

    def forward(self, x) -> torch.Tensor:
        """Creates equivariant pose prediction from observation

        Args:
            state (torch.Tensor): Observation of image with block in it
        Returns:
            q-value estimation associated with the state-action pair
        """
        assert x.shape[1:] == (self.input_shape,), f"Observation shape must be {self.input_shape}, current is {x.shape[1:]}"

        batch_size = x.shape[0]

        if hasattr(self, 'cnn'):
            x = self.cnn(x)

        mlp_out = self.mlp(x)

        return mlp_out

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        "Predicts 4d action for an input state, without storing gradient"
        V = self.forward(state)
        return V
    
    def update_baseline(self, s, a, r, sp, d):
        '''optimizes the value estimator
        '''

        

    def td_loss(self, s, a, r, sp, d):
        '''Calculates the td-loss for the value network based on a transition
        '''
        return self.mse(
                self.predict(s),
                r + self.gamma * self.predict(sp))

class QValueNetworkReturns(ValueNetworkReturns, nn.Module):

    def forward(self, state, action) -> torch.Tensor:
        assert state.shape[1:] == self.input_shape, f"Observation shape must be {self.input_shape}, current is {state.shape[1:]}"

        batch_size = state.shape[0]
        conv_out = self.cnn(state)

        hs = torch.cat((conv_out, action), 1)

        mlp_out = self.mlp(hs, 1)

        return mlp_out

    def __call__(self, state, action):
        return self.forward(state, action)

class AdvantageNetworkReturns(ValueNetworkReturns, nn.Module):
    pass

if __name__ == '__main__':
    net = QValueNetworkReturns(64, from_image=False)

    inp = torch.randn((1, 64))
    print(net(inp, 4))

