import numpy as np
import torch
from torch import nn
import gym
from torch.optim import Adam

from ...networks.feedforward import FeedForward

from . import AbstractPolicy

"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class DeterministicPolicy(AbstractPolicy, nn.Module):
    '''A determinisic Policy. States are mapped directly to actions at the output of the network, with no 
    '''

    def __init__(self,
            state_space,
            action_space,
            n_hidden=1,
            hidden_size=128, 
            lr:float=1e-3,
            noise=0.3,
            dropout=0.6,
            ):

        AbstractPolicy.__init__(self, state_space, action_space, lr)
        nn.Module.__init__(self)
        assert isinstance(action_space, gym.spaces.Box), 'Discrete action space cannot be continuous'

        self.mean = FeedForward(d_input=self.state_dim, d_output=self.action_dim, n_hidden=n_hidden, d_hidden=hidden_size, dropout=dropout)

        # Initialize a random process for exploration
        self.sd = noise
        self.noise = OUNoise(self.action_space, max_sigma=self.sd, min_sigma=self.sd)
        self.i = 0

        self.noise.reset()
        self.optim = Adam(self.get_params(), lr=self.lr )

    def pdf(self, state):
        return super().pdf(state)

    def score(self, s, a, v):
        return super().score(s, a, v)

    def __call__(self, state: np.ndarray):
        action = self.forward(state).detach().numpy()
        action = self.noise.get_action(action, self.i)
        return action
    
    def optimize(self, policy_loss:torch.Tensor):
        '''updates network parameters according to score calculation.
        '''
        self.i += 1
        self.optim.zero_grad()
        policy_loss.backward()
        self.optim.step()
        del policy_loss

    def forward(self, state):
        action = torch.tanh(self.mean(torch.FloatTensor(state)))
        return action 

    def get_params(self):
        return self.parameters()
