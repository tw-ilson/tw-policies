from utils import action_mask
import torch
from torch import nn
import numpy as np
from models.cnn import CNN
from models.returns import AbstractReturns
import gymnasium as gym
from gymnasium.spaces.utils import flatdim

from models.feedforward import FeedForward

class ValueNetworkReturns(AbstractReturns, torch.nn.Module):
    def __init__(self, state_space, n_hidden=1, hidden_size=64, lr=1e-3, gamma=0.98) -> None:
        """Produces value estimation for state-action pairs 

        Args:
            cnn: the feature extractor to apply to the image
            input_shape (tuple, optional): Shape of the image (C, H, W). Should be square.
            NOTES: have 2 predictions of direct joint values: (base and gripper), two predictions
            of combinations of motors: (forward z values, and vertical position adjust)
        """
        super(ValueNetworkReturns, self).__init__()
        # (B, C, H, W)

        assert state_space.is_np_flattenable, "needs to be flattened for Neural Network"
        self.gamma = gamma

        self.d_state = flatdim(state_space)

        self.feedforward = FeedForward(
                d_input=self.d_state,
                d_output=1,
                d_hidden=hidden_size,
                n_hidden=n_hidden)

        self.mse = torch.nn.MSELoss()
        self.optim = torch.optim.SGD(self.parameters(), lr=lr)

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state) -> torch.Tensor:
        """Creates equivariant pose prediction from observation

        Args:
            state (torch.Tensor): Observation of image with block in it
        Returns:
            q-value estimation associated with the state-action pair
        """
        state = torch.as_tensor(state)
        return self.feedforward(state).item()

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        "Predicts 4d action for an input state, without storing gradient"
        V = self.forward(state)
        return V
    
    def optimize(self, s, td_err):
        '''optimizes the value estimator
        '''
        self.optim.zero_grad()
        loss = td_err * self.forward(s)
        loss.backward()
        self.optim.step()

    def td_err(self, s, a, r, sp):
        '''Calculates the SARS td-err for the value network based on a transition
        '''
        return self.mse(
                self.predict(s),
                r + self.gamma * self.predict(sp))

class QValueNetworkReturns(nn.Module):
    """Q-value network which computes a predicted return based on current state and action taken
    """

    def __init__(self, state_space, action_space, n_hidden=1, hidden_size=64, lr=0.001, gamma=0.98) -> None:
        super(QValueNetworkReturns, self).__init__()
        assert state_space.is_np_flattenable, "state space needs to be flat for neural network"
        assert action_space.is_np_flattenable, "action space needs to be flat for neural network"

        self.gamma = gamma

        self.discrete_action = isinstance(action_space, gym.spaces.Discrete)

        self.d_state = flatdim(state_space)
        self.d_action = flatdim(action_space)

        self.feedforward = FeedForward(
                d_input= self.d_state + self.d_action,
                d_output=1,
                d_hidden=hidden_size,
                n_hidden=n_hidden)

        self.mse = torch.nn.MSELoss()
        self.optim = torch.optim.SGD(self.parameters(), lr=lr)
    def __call__(self, state, action):
        return self.predict(state, action)

    def forward(self, state, action) -> torch.Tensor:
        if self.discrete_action:
            action = action_mask(action, self.d_action)
        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        x = torch.cat((state, action), dim=1)
        return self.feedforward(x)

    @torch.no_grad()
    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        Q = self.forward(state, action)
        return Q.numpy()

    def optimize(self, s, a, td_err):
        '''optimizes the q-value estimator
        '''
        self.optim.zero_grad()
        loss = td_err * self.forward(s, a)
        loss.backward()
        self.optim.step()

    def td_err(self, s, a, r, sp, ap) -> float:
        '''Calculates the SARSA td-err for the value network based on a transition
        '''
        t1 = self.forward(s, a)
        t2 = torch.as_tensor(
                r + self.gamma * self.predict(sp, ap))
        return self.mse(t1, t2)

class AdvantageNetworkReturns(ValueNetworkReturns, nn.Module):
    def __init__(self, d_state, n_hidden=1, hidden_size=64, alpha=0.001, from_image=False) -> None:
        super().__init__(d_state, n_hidden, hidden_size, alpha, from_image)
        raise NotImplemented


