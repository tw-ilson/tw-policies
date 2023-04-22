import numpy as np
import torch
from torch import device, nn
import gym
from torch.optim import Adam

from ...networks import FeedForward
from ...utils import prepare_batch

from . import AbstractPolicy

class LinearGaussianPolicy(AbstractPolicy):
    '''Approximates a continuous policy using a Linear combination of the features of the state. Predicts a mean and standard deviation for each factor of the action space.
    '''
    def __init__(self, state_space, action_space, alpha) -> None:
        super().__init__(state_space, action_space, alpha, is_continuous=True)
        assert not isinstance(self.env.action_space, gym.spaces.Discrete), 'Discrete action space cannot be continuous'

        self.mu_weight = np.random.uniform(-1, 1, size=(self.action_dim, self.state_dim))
        self.sd_weight = np.ones(self.action_dim)

        self.EPS = 1e-6
    
    def forward(self, state):
        #Dense linear combination of state features
        mu = np.dot(self.mu_weight, state)

        #log-sd is based on pure weights
        sd = np.exp(self.sd_weight) #- logsumexp(self.sd_weight))
        sd = np.clip(sd, self.EPS, 2)

        return mu, sd

    def pdf(self, state):
        '''The gaussian log-probability density calculation
        '''
        mu, sd = self.forward(state)
        log_probs = lambda x: -0.5 * (x - mu)/sd**2 - np.log(sd * (2 * np.pi)**(0.5))
        return log_probs

    def __call__(self, state: np.ndarray):
        #sample randomly from gaussian distribution
        mu, sd = self.forward(state)
        dev = np.random.multivariate_normal(mu, np.diag(sd))

        act = mu + dev
        
        assert len(act) == self.action_dim
        return act

    def score(self, s, a, v):
        ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
            which follows a single linear combination layer of the state input.
        '''
        def step_score(s_t, a_t, v_t):
            mu, sd = self.forward(s_t)

            mu_grad = v_t * np.outer(s_t, (a_t - mu))/(sd**2)
            sd_grad = ((a_t - mu)**2 - sd**2)/ sd**3
            return np.array([mu_grad, sd_grad])
        
        scores = np.array([step_score(s_t, a_t, v_t) for s_t, a_t, v_t in zip(s, a, v)])
        return np.mean(scores[:, 0], axis=0), np.mean(scores[:, 1], axis=0)

    def optimize(self, score):
        #optimize using computed gradients
        self.mu_weight = self.mu_weight + self.lr * score[0].T
        self.sd_weight = self.sd_weight + self.lr * score[1]

        self.mu_weight = np.nan_to_num(self.mu_weight)
        self.sd_weight = np.nan_to_num(self.sd_weight)

    def get_params(self):
        return [self.mu_weight, self.sd_weight]

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
