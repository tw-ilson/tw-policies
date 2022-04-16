from typing import Any, Tuple, List
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from cnn import CNN


class AbstractReturns(ABC):
    """ A policy gradient is only useful in the context of the state from which it was produced, and more generally, the utility of the playout from which it came, in terms of real success or failure. We define this data structure in order to conceptualize and compare different approaches to the problem of advantage estimation, as we get into more sophisticated policy gradient methods.
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        self.debug = {}
        super().__init__()
        
    @abstractmethod
    def __call__(self, **kwargs):
        pass

class MonteCarloReturns(AbstractReturns):
    """ This advantage function requires full monte-carlo style playouts of a policy in order to directly calculate the returns from each state in the given playout.
    """

    def __init__(self, gamma) -> None:
        super().__init__()
        self.gamma = gamma

    def __call__(self, playout) -> dict:
        """ Given a full playout, implements top-down dynamic programming to compute the return following from each state encountered in the trajectory
        """
        trajectory = OrderedDict()
        t = len(playout) - 1
        s_t, a,  reward, log_prob = playout[-1]
        trajectory[bytes(s_t)] = reward
        while t > 0:
            t-=1
            s_t, a, reward, log_prob = playout[t]
            ret_tnext = trajectory[bytes(playout[t+1][0])]
            ret_t = reward + self.gamma * ret_tnext
            trajectory[bytes(s_t)] = ret_t
            assert not np.isnan(ret_t)

        return trajectory

class MonteCarloBaselineReturns(MonteCarloReturns):
    """ This Advantage function requires full monte-carlo style playouts of a policy in order to directly calculate the returns from each state in the given playout.
    Here we introduce the notion of a baseline, which can be any function, that we subtract our returns 
    """

    def __init__(self, gamma, baseline) -> None:
        super().__init__(gamma)
        self.baseline = baseline

    def __call__(self, playout) -> dict:
        """ Given a full playout, implements top-down dynamic programming to compute the return following from each state encountered in the trajectory
        """
        traj_ret = super.__call__(playout)
        traj_b = {bytes(s): self.baseline(s) for s, a, r, p in playout}

        return {s: traj_ret[s] - traj_b[s] for s in traj_ret}

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
            self.cnn = CNN()
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
    
    def optimize(self, s, a, r, sp, d):
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

if __name__ == '__main__':
    net = QValueNetworkReturns(64, from_image=False)

    inp = torch.randn((1, 64))
    print(net(inp, 4))

