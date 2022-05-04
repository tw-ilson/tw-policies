from abc import ABC, abstractmethod
from typing import Tuple

import gym.spaces.utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from cnn import CNN
from utils import logsumexp, prepare_batch

class AbstractPolicy(ABC):
    '''Abstract class to represent a stochastic policy function approximator.
    '''

    def __init__(self, state_space, action_space, alpha) -> None:
        print(f"State space: {state_space}\nAction space: {action_space}")
        self.state_space = state_space
        self.state_dim = gym.spaces.utils.flatdim(state_space)
        self.action_space = action_space
        self.action_dim = gym.spaces.utils.flatdim(action_space)
        self.alpha = alpha 

    @abstractmethod
    def pdf(self, state):
        '''Returns the log probability density function of this policy distribution
        '''
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def score(s, a, v):
        ''' Given a full batch of transition using this policy, computes the expectation of the score gradient scaled by the advantage
        Params
        ------
            s: states (array-like)
            a: actions (array-like)
            v: value/advantages (array-like)

        '''
        pass

    @abstractmethod
    def optimize(self, score) -> None:
        '''  Performs backwards pass to updates network parameters according to score calculation.
        Params
        ------
            score: the 'score function' as described in PG literature, typically computed using log probability of action scaled by 
            advantage: the advantage of the state or sequence from which the score was computed.
        '''
        pass

    @abstractmethod
    def __call__(self, state:np.ndarray):
        '''Given a state, return an action and an approximation of the log probability vector of the action space distribution
        '''
        pass

class LinearDiscretePolicy(AbstractPolicy):
    '''Approximates a policy using a Linear combination of the features of the state'''

    def __init__(self, state_dim, action_space, alpha) -> None:
        super().__init__(state_dim, action_space, alpha)
        self.state_dim = self.state_dim

        #initialize weights matrix, single fully connected layer, no bias 
        self.weights = np.random.uniform(0, 1, size=(self.action_dim, self.state_dim)).astype(np.float64)
        self.d_weights = self.weights.T.copy() * 0

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
            def step_score(s_t, a_t):
                assert len(s_t) == self.state_dim
                sm_probs = self.pdf(s_t)
                
                #one-hot encode action
                act_mask = np.zeros(self.action_dim)
                act_mask[a_t] = 1

                phi = np.outer(s_t, np.ones(self.action_dim))
                return phi - np.nan_to_num(
                        np.outer(s_t, np.exp(sm_probs * act_mask)))

            score = np.sum([v_t * step_score(s_t, a_t) for s_t, a_t, v_t in zip(s, a, v)], axis=0)
            return score


    def optimize(self, score):
        #optimize using computed gradients
        
        self.weights = self.weights + self.alpha * score.T
        self.d_weights = np.zeros((self.state_dim, self.action_dim))

    def get_params(self):
        return self.weights
    
class LinearGaussianPolicy(AbstractPolicy):
    '''Approximates a continuous policy using a Linear combination of the features of the state. Predicts a mean and standard deviation for each factor of the action space.
    '''
    def __init__(self, state_space, action_space, alpha) -> None:
        super().__init__(state_space, action_space, alpha)
        assert isinstance(state_space, gym.spaces.Box)

        self.mu_weight = np.random.uniform(-1, 1, size=(self.action_dim, self.state_dim))
        self.sd_weight = np.ones(self.action_dim)

        # self.d_mu_weight = self.mu_weight.T.copy() * 0
        # self.d_sd_weight = self.sd_weight.T.copy() * 0

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
        self.mu_weight = self.mu_weight + self.alpha * score[0].T
        self.sd_weight = self.sd_weight + self.alpha * score[1]

        self.mu_weight = np.nan_to_num(self.mu_weight)
        self.sd_weight = np.nan_to_num(self.sd_weight)

    def get_params(self):
        return [self.mu_weight, self.sd_weight]

class NNDiscretePolicy(AbstractPolicy, nn.Module):
    '''Neural Network approximator for a categorical policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            hidden_size=64, 
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicy.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)

        if from_image:
            self.state_dim = self.state_space.shape
            self.conv = CNN(self.state_dim)
            self.layers = nn.Sequential(
                    self.conv,
                    nn.Linear(self.conv.output_size, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_dim),
                    )
        else:
            self.layers = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_dim),
                    )

        self.optim = Adam(self.get_params(), lr=self.alpha )

    def pdf(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        return dist

    def __call__(self, state: np.ndarray):
        dist = self.pdf(state)
        action = dist.rsample()
        return action.item()
        
    def score(self, s, a, v):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        s, a, v = prepare_batch(s, a, v)
        v = torch.tensor(v)
        dist = self.pdf(s)
        return torch.mean(-dist.log_prob(torch.tensor(a)) * v)
    
    def optimize(self, score):
        '''updates network parameters according to score calculation.
        '''
        score.backward(retain_graph=True)
        #scale gradient by advantage 
        self.optim.step()
        self.optim.zero_grad()
        score.detach_()

    def forward(self, state):

        x = self.layers(torch.FloatTensor(state))
        y = F.softmax(x, dim=-1)
        return y

    def get_params(self):
        return self.parameters()

class NNGaussianPolicy(AbstractPolicy, nn.Module):
    '''Neural Network approximator for a continuous policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            hidden_size=64, 
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicy.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)

        if from_image:
            self.state_dim = self.state_space.shape
            self.conv = CNN(self.state_dim)
            self.mu = nn.Sequential(
                    self.conv,
                    nn.Linear(self.conv.output_size, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_dim),
                    )
        else:
            self.mu = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_dim)
                    )

        # self.sigma = nn.Linear(hidden_size, self.action_dim)
        self.sigma = torch.ones(self.action_dim, dtype=torch.float32, requires_grad=True)

        self.optim = Adam(list(self.get_params()) + [self.sigma], lr=self.alpha )

    def pdf(self, state):
        mu, sd = self.forward(torch.tensor(state))
        dist = torch.distributions.Normal(mu, sd)
        return dist

    def __call__(self, state: np.ndarray):
        dist = self.pdf(state)
        action = dist.rsample()
        return action.detach().numpy()
    
    def score(self, s, a, v):
        ''' Computes the score function of the policy gradient with respect to the parameters
        '''
        states, actions, values = prepare_batch(s, a, v)
        dist = self.pdf(states)
        return torch.mean(-dist.log_prob(actions) * values)
    
    def optimize(self, score):
        '''updates network parameters according to score calculation.
        '''
        score.backward(retain_graph=True)
        self.optim.step()
        self.optim.zero_grad()
        score.detach_()

    def forward(self, state):
        EPS = 1e-6
        MAX = 3
        return self.mu(torch.FloatTensor(state)), torch.clip(torch.exp(self.sigma), EPS, MAX)

    def get_params(self):
        return self.parameters()

if __name__ == '__main__':
    import gym
    env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('Pendulum-v1')
    env.reset()
    pol = LinearDiscretePolicy(env.observation_space, env.action_space, 1e-3)
    s = np.ones((10, pol.state_dim))
    a = np.ones(10, dtype=int)
    r = np.ones(10)
    print(pol.score(s, a, r))

