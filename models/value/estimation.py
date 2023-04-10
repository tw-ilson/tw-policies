import torch
from torch import nn
import numpy as np
import  gym
from gym.spaces.utils import flatdim

from .montecarlo import MonteCarloReturns

from ...networks import FeedForward, CNN
from ...utils import action_mask, prepare_batch
from . import AbstractReturns
from torch.nn.functional import mse_loss

def TDErr(v, vp, r, discount):
    ''' time-difference error between numeric value at t, and value at the following timestep
    '''
    v = torch.as_tensor(v).squeeze()
    vp = torch.as_tensor(vp).squeeze()
    v_target = r + discount * vp
    return mse_loss(v, v_target)

class ValueNetworkReturns(AbstractReturns, torch.nn.Module):
    def __init__(self, state_space, n_hidden=1, hidden_size=64, lr=1e-3, gamma=0.98) -> None:
        """Produces value estimation for state-action pairs 

        Args:
            input_shape (tuple, optional): Shape of the image (C, H, W). Should be square.
            NOTES: have 2 predictions of direct joint values: (base and gripper), two predictions
            of combinations of motors: (forward z values, and vertical position adjust)
        """
        super(ValueNetworkReturns, self).__init__()

        # assert state_space.is_np_flattenable, "needs to be flattened for Neural Network"
        self.gamma = gamma

        self.d_state = flatdim(state_space)

        self.feedforward = FeedForward(
                d_input=self.d_state,
                d_output=1,
                d_hidden=hidden_size,
                n_hidden=n_hidden)

        self.mse = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

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
        return self.feedforward(state)

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        "Predicts 4d action for an input state, without storing gradient"
        V = self.forward(state)
        return V
    
    def optimize(self, s, td_err):
        '''optimizes the value estimator
        '''
        self.optim.zero_grad()
        loss = td_err 
        loss.backward()
        self.optim.step()

    def td_err(self, s, a, r, sp):
        '''Calculates the SARS td-err for the value network based on a transition
        '''
        return TDErr(self.predict(s), self.predict(sp), r, self.gamma)

class QValueNetworkReturns(AbstractReturns, nn.Module):
    """Q-value network which computes a predicted return based on current state and action taken
    """

    def __init__(self, state_space, action_space, n_hidden=1, hidden_size=64, lr=0.001, gamma=0.98) -> None:
        super(QValueNetworkReturns, self).__init__()
        # assert state_space.is_np_flattenable, "state space needs to be flat for neural network"
        # assert action_space.is_np_flattenable, "action space needs to be flat for neural network"

        self.gamma = gamma

        self.discrete_action = isinstance(action_space, gym.spaces.Discrete)

        self.d_state = flatdim(state_space)
        self.d_action = flatdim(action_space)

        self.feedforward = FeedForward(
                d_input= self.d_state + self.d_action,
                d_output=1,
                d_hidden=hidden_size,
                n_hidden=n_hidden)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def __call__(self, state, action):
        return self.forward(state, action)

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
        loss = td_err #* self.forward(s, a)
        loss.mean().backward()
        self.optim.step()
        del loss

    def td_err(self, s, a, r, sp, ap) -> float:
        '''Alias function; Calculates the SARSA td-err for the value network based on a transition
        '''
        return TDErr(self.predict(s, a), self.predict(sp, ap), r, self.gamma)

class AdvantageNetworkReturns(AbstractReturns, nn.Module):
    """ In the advantage estimation, we are calculating the difference between the reward + Q-value (state-action value) and the Value (state value). This represents the amount of 'advantage' in taking a *specific* action at a state. AKA the estimated TD-residual
    """ 
    def __init__(self, state_space, n_hidden=1, hidden_size=64, lr=1e-5, gamma=0.98) -> None:
        super(AdvantageNetworkReturns, self).__init__()
        self.V = ValueNetworkReturns(state_space, n_hidden, hidden_size, lr, gamma)
        self.entropy = 0

    def __call__(self, r, s, sp):
        return self.forward(r, s, sp)

    def forward(self, r, s, sp):
        r, s, sp = prepare_batch(r, s, sp)
        v = self.V.forward(s).permute(1,0).squeeze()
        vp = self.V.forward(sp).permute(1,0).squeeze()
        out = vp + r - v
        return out

    def optimize(self, advantage, log_prob):
        """ The network output is already an error messor, so loss is simple mse loss
        """
        self.V.optim.zero_grad()
        self.entropy += float(log_prob)
        loss = 0.5 * advantage.pow(2) + 0.001 * self.entropy
        loss.mean().backward()
        self.V.optim.step()
        del advantage
        del loss
        # return loss

class GAEReturns(AbstractReturns, nn.Module):
    """ Generalized Advantage Estimator
    """

    def __init__(self, gae_param, state_space, n_hidden=1, hidden_size=64, lr=1e-5, gamma=0.98) -> None:
        """ Lambda is the hyperparameter introduced by GAE, to facilitate the tradeoff between the actual, high-variance rewards, and the estimated value function.
        """
        super(GAEReturns, self).__init__()
        self.gae_param = gae_param
        self.A = AdvantageNetworkReturns(state_space, n_hidden, hidden_size, lr, gamma)

    def __call__(self, states, rewards):
        return self.forward(states, rewards)

    def forward(self, states, rewards):
        total = 0
        for l, r in enumerate(rewards):
            total += (self.gamma * self.gae_param)**l * self.A(r, states[l], states[l+1])
