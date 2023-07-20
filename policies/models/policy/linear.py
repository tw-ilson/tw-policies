"""
Linear function approximators using only numpy
"""
from . import AbstractPolicy
from utils import logsumexp, action_mask
from torch import is_tensor
import gym.spaces
import numpy as np

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
            if is_tensor(v):
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
