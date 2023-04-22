import numpy as np
import torch
from torch import is_tensor, nn
from torch.distributions import Categorical

from . import AbstractPolicy
from ...networks import FeedForward, CNN
from ...utils import logsumexp, prepare_batch, action_mask

class LinearDiscretePolicy(AbstractPolicy):
    '''Approximates a policy using a Linear combination of the features of the state'''

    def __init__(self, state_space, action_space, alpha) -> None:
        super().__init__(state_space, action_space, alpha)
        #initialize weights matrix, single fully connected layer, no bias 
        self.weights = np.random.uniform(0, 1, size=(self.action_dim, self.state_dim)).astype(np.float64)
        # self.d_weights = self.weights.T.copy() * 0

    def pdf(self, state):
        ''' The Log-Softmax produces categorical distribution
        '''
        x = self.forward(state)
        assert not all(i == 0 for i in x)
        log_probs = x - logsumexp(x)
        return log_probs
    
    def forward(self, state):
        #Dense linear combination of state features
        return np.dot(self.weights, state)

    def __call__(self, state: np.ndarray):
        log_probs = self.pdf(state)

        #convert back from log space to discrete categorical using Gumbel-max trick:
        g = np.random.gumbel(0, .99, size=self.action_dim)
        action = np.argmax(log_probs + g) #sample

        return action

    def score(self, s, a, v):
            ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
                which follows a single linear combination layer of the state input.
            '''
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            def step_score(s_t, a_t, v_t):
                assert len(s_t) == self.state_dim
                sm_probs = self.pdf(s_t)
                #one-hot encode action
                act_mask = action_mask(a_t, self.action_dim)
                phi = np.outer(s_t, np.ones(self.action_dim))
                # Cross-Entropy loss
                return v_t * (phi - np.nan_to_num(
                        np.outer(s_t, np.exp(sm_probs * act_mask))))
            # score = np.sum([step_score(s_t, a_t, v_t) 
            #                 for s_t, a_t, v_t in zip(s, a, v)], axis=0)
            scores = np.array([step_score(s_t, a_t, v_t) for s_t, a_t, v_t in zip(s, a, v)])
            return scores.sum(axis=0) # THIS IS ACTUALLY THE GRADIENT OF SCORE


    def optimize(self, score):
        #optimize using computed gradients
        self.weights = self.weights + self.lr * score.T

    def get_params(self):
        return self.weights

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
