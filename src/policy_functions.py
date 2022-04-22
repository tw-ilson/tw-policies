from abc import ABC, abstractmethod
from typing import Tuple

from gym.spaces.utils import flatdim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.distributions import Categorical, Normal
from cnn import CNN


class AbstractPolicyApproximator(ABC):
    '''Abstract class to represent a stochastic policy function approximator.
    '''

    def __init__(self, state_space, action_space, alpha) -> None:
        self.state_space = state_space
        self.state_dim = ...
        self.action_space = action_space
        self.action_dim = ...
        self.alpha = alpha
        print(f"state space: {state_space}\naction space: {action_space}")

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def episode_score(self, tau):
        ''' Given a full playout of this policy, computes the score sum for the loss output of the policy approximation
        '''
        pass

    @abstractmethod
    def step_score(self, s_t, a_t):
        ''' Given a full playout of this policy, computes the score sum for the loss output of the policy approximation
        '''
        pass

    @abstractmethod
    def update_gradient(self, state, score, advantage):
        '''Given a score and an advantage value, computes the gradient of the approximator wrt the weights
        '''
        pass

    @abstractmethod
    def optimize(self):
        '''  Performs backwards pass to updates network parameters according to score calculation.
        Params
        ------
            score: the 'score function' as described in PG literature, typically computed using log probability of action scaled by 
        '''
        pass

    @abstractmethod
    def __call__(self, state):
        '''Given a state, return an action and an approximation of the log probability vector of the action space distribution
        '''
        pass

class LinearDiscretePolicyApproximator(AbstractPolicyApproximator):
    '''Approximates a policy using a Linear combination of the features of the state'''

    def __init__(self, state_space, action_space, alpha) -> None:
        super().__init__(state_space, action_space, alpha)
        self.state_dim = flatdim(state_space)
        self.action_dim = flatdim(action_space)

        #initialize weights matrix, single fully connected layer, no bias 
        self.weights = np.random.normal(0, 1, size=(self.action_dim, self.state_dim)).astype(np.float64)

        self.d_weights = np.zeros((self.state_dim, self.action_dim))
        
    
    def forward(self, state):
        #Dense linear combination of state features
        hs = np.dot(self.weights, state)

        #SoftMax following output layer; numerically stable
        def log_softmax(x):
            assert not any([i == 0 for i in x])
            max_x = np.max(x)
            log_probs = x - max_x - np.log(np.sum(np.exp(x - max_x)))
            return log_probs

        #take the softmax function over hidden states
        log_probs = log_softmax(hs)

        return log_probs

    def __call__(self, state: np.ndarray):
        '''Computes the forward pass and samples from the output distribution
        '''

        log_probs = self.forward(state)

        #convert back from log space to discrete categorical using Gumbel-max trick:
        g = np.random.gumbel(0, .99, size=self.action_dim)
        action = np.argmax(log_probs + g) #sample
        return action 

    def episode_score(self, tau):
        '''computes score with respect to whole playout
        '''
        score = 0
        for s_t, a_t, r in tau:
            score += self.step_score(s_t, a_t)
        return score

    def step_score(self, s_t, a_t):
        ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
            which follows a single linear combination layer of the state input.
        '''
        
        # probs = self.forward(s_t)
        # hs = np.dot(self.weights, s_t) 
        # temp = hs - sum(probs * hs)
        #
        # #one-hot encode action
        # act_mask = np.zeros(self.action_dim)
        # act_mask[a_t] = 1
        #
        # score = temp * act_mask
        #
        #
        # return score
        sm_probs = self.forward(s_t)

        #one-hot encode action
        act_mask = np.zeros(self.action_dim)
        act_mask[a_t] = 1

        #the score computation
        score = -sm_probs[a_t]

        #The components of the gradient via chain rule
        SM =  sm_probs.reshape((-1,1))
        Jsm_dh = np.diagflat(sm_probs) - np.dot(SM, SM.T)
        # dh_dw = s_t

        dL_dh = np.dot(Jsm_dh, act_mask) / score
        return dL_dh

    def update_gradient(self, state, score, advantage):
        #backward pass
        #gradient of cross-entropy wrt weights:
        self.d_weights += np.outer(state, advantage * score)
        assert not np.isnan(self.d_weights).any()

    def optimize(self):
        #optimize using computed gradients

        self.weights = self.weights + self.alpha * self.d_weights.T
        self.d_weights = np.zeros((self.state_dim, self.action_dim))

    def get_params(self):
        return self.weights
    

class NNDiscretePolicyApproximator(AbstractPolicyApproximator, nn.Module):
    '''Neural Network approximator for a categorical policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            hidden_size=64, 
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicyApproximator.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)

        self.action_dim = flatdim(self.action_space)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

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
            self.state_dim = flatdim(self.state_space)
            self.layers = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_dim))

        self.optim = Adam(self.get_params(), lr=self.alpha )
        self.to(self.device)

    def __call__(self, state: np.ndarray) -> Tuple[int, torch.tensor]:
        probs = self.forward(torch.tensor(state).unsqueeze(0))
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
        
    def episode_score(self, tau, G_tau):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        scores = []
        for s_t, a_t, r, log_prob in tau:
            score = self.step_score(s_t, a_t, r, log_prob, G_tau(s_t))
            score.backward()
            scores.append(score.unsqueeze(0))
        # torch.cat(scores).sum()

    def step_score(self, s_t, a_t):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        return -log_prob * G
    
    def optimize(self, score):
        '''updates network parameters according to score calculation.
        '''
        # score.to(self.device)
        # score.backward(retain_graph=True)
        self.optim.step()
        self.optim.zero_grad()

        #inplace detach so we can plot this
        score.detach_().cpu()

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32, device='cpu')
        x = self.layers(state)
        y = F.softmax(x, dim=-1).cpu()
        return y

    def get_params(self):
        return self.parameters()

class LinearGaussianPolicyApproximator(AbstractPolicyApproximator):
    '''Linear regression gaussian policy 
    '''

    def __init__(self, state_space, action_space, alpha, fixed_variance=None) -> None:
        super().__init__(state_space, action_space, alpha)
        self.state_dim = flatdim(state_space)
        self.action_dim = flatdim(action_space)


        #initialize weights matrix, single fully connected layer, no bias 
        if fixed_variance:
            self.mu_weights = np.random.uniform(0, 1, size=(self.action_dim, self.state_dim))
            self.variance = fixed_variance
            self.d_mu_weights = self.mu_weights.copy() * 0
        else:
            self.mu_weights = np.random.uniform(0, 1, size=(2*self.action_dim, self.state_dim))
            self.var_weights = np.random.uniform(0, 1, size=(2*self.action_dim, self.state_dim))
            self.d_mu_weights = self.mu_weights.copy() * 0
            self.d_var_weights = self.var_weights.copy() * 0
    
    def forward(self, state):
        #Dense linear combination of state features
        mu = np.dot(self.mu_weights, state)

        #sample randomly from gaussian distribution

        if self.variance:
            var = self.variance
            act = np.random.normal(mu, self.variance, )
        else:
            var = np.dot(self.var_weights, state)
            act = np.random.normal(mu, var)

        return act, mu, var

    def __call__(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        pass

    def episode_score(self, tau, G_tau):
        '''computes score with respect to whole playout
        '''

        score = 0
        for s_t, a_t, r, log_prob in tau:
            score += self.step_score(s_t, a_t, r, log_prob, G_tau(s_t))
        return score

    def step_score(self, s_t, a_t, r, log_prob, G):
            ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
                which follows a single linear combination layer of the state input.
            '''
            sm_probs = self.forward(s_t)

            #one-hot encode action
            act_mask = np.zeros(self.action_dim)
            act_mask[a_t] = 1

            #the score computation
            score = -log_prob
            
            # #The components of the gradient via chain rule
            # SM =  sm_probs.reshape((-1,1))
            # Jsm_dh = np.diagflat(sm_probs) - np.dot(SM, SM.T)
            dh_dw = s_t
            #
            # #gradient descent step 
            # dL_dh = -G * np.dot(Jsm_dh, act_mask) / log_prob
            dL_dh = act_mask * -log_prob * G
            self.d_weights += np.outer(dh_dw, dL_dh)

            assert not np.isnan(self.d_weights).any()
            return -log_prob * G


    def optimize(self, score):
        #optimize using computed gradients
        
        self.weights = self.weights + self.alpha * self.d_weights.T
        self.d_weights = np.zeros((self.state_dim, self.action_dim))

    def get_params(self):
        return self.weights

class NNGaussianPolicyApproximator(AbstractPolicyApproximator, nn.Module):
    '''Neural Network approximator for a continuous policy distribution. As opposed to the softmax distribution on the output of the Discrete approximator, this policy samples from a gaussian normal distribution 
    '''

    def __init__(self,
            state_space,
            action_space,
            hidden_size=32, 
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicyApproximator.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)


        if torch.cuda.is_available():
            self.device = 'cuda'

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
            self.state_dim = flatdim(self.state_space)

            self.layers = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(True),  
                    )

        self.mu_out = nn.Linear(hidden_size, self.action_dim)
        self.var_out = nn.Linear(hidden_size, self.action_dim)


        self.optim = Adam(self.get_params(), lr=self.alpha )

    def __call__(self, state: np.ndarray):
        mu, var = self.forward(state)
        dist = Normal(mu, var)
        return dist.sample(), mu, var

        
        
    def episode_score(self, s_t, a_t, log_prob, advantage):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        score = -log_prob * advantage
        # print(score)
        return score.unsqueeze(0)

    def compute_gradient(self, state, score, advantage):
        pass
    
    def optimize(self):
        '''updates network parameters according to score calculation.
        '''
        # score = torch.cat(scores).sum()
        # score.backward(retain_graph=True)
        self.optim.step()
        self.optim.zero_grad()

    def forward(self, state):
        h = self.layers(torch.FloatTensor(state))
        return self.mu_out(h), self.var_out(h)

    def get_params(self):
        return self.parameters()
