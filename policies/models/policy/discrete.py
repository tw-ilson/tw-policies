import numpy as np
import torch
from torch import is_tensor, nn
from torch.distributions import Categorical

from . import AbstractPolicy
from ...networks import FeedForward, CNN
from ...utils import logsumexp, prepare_batch, action_mask


class NNDiscretePolicy(AbstractPolicy, nn.Module):
    '''Neural Network approximator for a categorical policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            n_hidden=1,
            hidden_size=32, 
            lr:float=1e-3,
            dropout:float=0.6
            ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        AbstractPolicy.__init__(self, state_space, action_space, lr)
        nn.Module.__init__(self)

        self.layers = FeedForward(
                self.state_dim,
                self.action_dim,
                n_hidden=n_hidden,
                d_hidden=hidden_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.optim = torch.optim.Adam(self.get_params(), lr=self.lr)
        self.to(self.device)

    def forward(self, state):
        
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        assert not torch.isnan(s).any()
        x = self.layers(s)
        logits = self.softmax(x)
        return logits

    def pdf(self, state):
        probs = self.forward(state)
        return Categorical(logits=probs)

    def __call__(self, state: np.ndarray):
        dist = self.pdf(state)
        action = dist.sample()
        return action.item()
        
    def score(self, s, a, v):
        ''' Computes the negative-log-likelihood of a chosen action, scaled by the value function at this state.
        '''
        s, a, v = prepare_batch(s, a, v)
        dist = self.pdf(s)

        #negative-log-likelihood 
        score = torch.sum(-dist.log_prob(a) * v.detach())
        return score
    
    def optimize(self, score):
        '''updates network parameters according to score calculation.
        '''
        torch.autograd.set_detect_anomaly(True)
        self.optim.zero_grad()
        score.backward()
        self.optim.step()
        score.detach_()

    def get_params(self):
        return self.parameters()
