import numpy as np
import torch
from torch import device, nn
import gym
from torch.optim import Adam

from ...networks import FeedForward
from ...utils import prepare_batch

from . import AbstractPolicy


class NNGaussianPolicy(AbstractPolicy, nn.Module):
    '''Neural Network approximator for a continuous policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            n_hidden=1,
            hidden_size=64, 
            lr:float=1e-3,
            annealing=False
            ):

        AbstractPolicy.__init__(self, state_space, action_space, lr, is_continuous=True)
        nn.Module.__init__(self)
        self.annealing = annealing
        assert not isinstance(action_space, gym.spaces.Discrete), 'Discrete action space cannot be continuous'
        self.mean = FeedForward(d_input=self.state_dim, 
                                d_output=self.action_dim, 
                                n_hidden=n_hidden, 
                                d_hidden=hidden_size, 
                                )
        # Homo-schedastic noise:
        self.sd = nn.Parameter(torch.ones(self.action_dim, dtype=torch.float32, requires_grad=True))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.optim = Adam(self.get_params(), lr=self.lr)
        self.to(self.device)

    def pdf(self, state):
        mu, sd = self.forward(state)
        return torch.distributions.MultivariateNormal(mu, sd)

    def forward(self, state):
        EPS = 1e-6
        MAX = 3
        state = torch.as_tensor(state, device=self.device)
        mu = torch.nn.functional.tanh(self.mean(state))
        torch.clip(torch.exp(self.sd), EPS, MAX)
        sigma = torch.diag_embed(self.sd)
        return mu, sigma 

    def __call__(self, state: np.ndarray):
        dist = self.pdf(state)
        action = dist.rsample()
        return action.cpu().detach().numpy()
    
    def score(self, s, a, v):
        ''' Computes the score function of the policy gradient with respect to the parameters
        '''
        states, actions, values = prepare_batch(s, a, v)
        dist = self.pdf(states)
        return torch.mean(-dist.log_prob(actions) * values)
    
    def optimize(self, score):
        '''updates network parameters according to score calculation.
        '''
        score.backward()
        self.optim.step()
        self.optim.zero_grad()
        score.detach_()

    def get_params(self):
        return self.parameters()
