from typing import List, Tuple

import abc
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from gym.spaces import Space, Discrete

# def compute_score(model, s_t, a_t, A): ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; updates network parameters according to score calculation.
#     Params
#     ------
#         s_t: state Tensor used as input to this policy (B, *state_shape)
#         a_t: index of action taken (B)
#         A: advantage multiplier (B)
#     Returns
#     ------
#         score: gradient of log policy probability weighted by advantage
#     '''
#
#     # get policy distribution of action (log probabilities)
#     a_dist = model.forward(s_t)
#     a_mask = torch.zeros(a_dist.shape)
#     a_mask[a_t] = A
#     loss = torch.sum(a_dist * a_mask, dim=-1)
#     a_mask.backward
#     loss.backward()
#     return loss

class AbstractPolicyApproximator(abc.ABC):
    '''Abstract class to represent a stochastic policy function approximator.
    '''

    def __init__(self, state_dim, action_space, alpha) -> None:
        self.state_dim = state_dim.shape[0]
        self.action_space = action_space.n
        self.alpha = alpha
        pass

    @abc.abstractmethod
    def get_params(self):
        pass

    @abc.abstractmethod
    def compute_score(self, s_t, a_t, log_prob, advantage):
        pass

    @abc.abstractmethod
    def update_params(self, score):
        '''  Performs backwards pass to updates network parameters according to score calculation.
        Params
        ------
            score: the 'score function' as described in PG literature, typically computed using log probability of action scaled by 
    '''

    @abc.abstractmethod
    def __call__(self, state:np.ndarray) -> np.ndarray:
        '''Given a state, return an action and an approximation of the log probability vector of the action space distribution
        '''
        pass

        

class CNN(nn.Module):
    ''' Convolutional half module of Actor network to predict actions from state '''

    def __init__(self):
        '''
        Input:
            84x84 grayscale image
        '''
        super().__init__()

        self.img_shape = 84

        self.sequence = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=2),
                nn.ReLU(True),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, kernel_size=4, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, kernel_size=4, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, kernel_size=4, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
            )

        self.output_size = self.compute_output_size(self.img_shape)

    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D grayscale float vector """
        def rgb2gray(I):
            return np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])

        I = I[35:195] # crop
        I = rgb2gray(I.astype(np.float32))
        I = I[::2,::2,0] # downsample by factor of 2

        return I.ravel()

    def forward(self, x):
        return self.sequence(x)

    def compute_output_size(self, img_shape) -> int:
        x = torch.zeros( img_shape, dtype=torch.float32)

        # add batch dimension
        x = x.unsqueeze(0)

        out = torch.flatten(self.forward(x))
        return out.shape[1]

class LinearPolicyApproximator(AbstractPolicyApproximator):
    '''Approximates a policy using a Linear combination of the features of the state'''

    def __init__(self, state_dim, action_space, alpha) -> None:
        super().__init__(state_dim, action_space, alpha)

        #initialize weights matrix, single fully connected layer, no bias 
        self.weights = np.random.uniform(-1, 1, size=(self.action_space, self.state_dim)).astype(np.float64)

        self.d_weights = np.zeros((self.state_dim, self.action_space))

    def __call__(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        #SoftMax following output layer
        def log_softmax(x):
            max_x = np.max(x)
            exp_x = np.exp(x - max_x)
            sum_exp_x = np.sum(exp_x)
            log_sum_exp_x = np.log(sum_exp_x)
            max_plus_log_sum_exp_x = max_x + log_sum_exp_x
            log_probs = x - max_plus_log_sum_exp_x
            return log_probs

        x = np.dot(self.weights, state)
        log_probs = log_softmax(x)
        probs = np.exp(log_probs)
        action =np.random.choice(range(self.action_space), p=probs)
        return action, log_probs[action]

    def compute_score(self, s_t, a_t, log_probs, advantage):
        ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
            which follows a single linear combination layer of the state input.
        '''
        sm_out = np.exp(log_probs)

        #one-hot encode action
        act_mask = np.zeros(self.action_space)
        act_mask[a_t] = 1 * advantage

        #The components of the gradient via chain rule
        dL_dsm = - sm_out ** -1
        SM =  sm_out.reshape((-1,1))
        Jsm_dx = np.diagflat(sm_out) - np.dot(SM, SM.T)
        dx_dw = s_t

        #gradient descent step 
        self.d_weights += np.outer(dx_dw, dL_dsm * np.dot(Jsm_dx, act_mask)).T

        return log_prob * advantage


    def update_params(self, score):
        #optimize using computed gradients
        self.weights = self.weights + self.alpha * self.d_weights
        self.d_weights = np.zeros

    def get_params(self):
        return self.weights
    

class NNPolicyApproximator(AbstractPolicyApproximator, nn.Module):
    '''The full policy network composed of convolutional and MLP parts '''

    def __init__(self,
            state_space:Space,
            action_space:Space,
            hidden_size=32, 
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicyApproximator.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)

        if not from_image:
            self.layers = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_space),
                )
        else:
            # self.layers.append(CNN())
            raise NotImplemented

        self.optim = Adam(self.get_params(), lr=self.alpha )

    def __call__(self, state: np.ndarray) -> Tuple[int, torch.tensor]:
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
        
    def compute_score(self, s_t, a_t, log_prob, advantage):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        score = -log_prob * advantage
        # print(score)
        return score.unsqueeze(0)
    
    def update_params(self, scores):
        '''updates network parameters according to score calculation.
        '''
        score = torch.cat(scores).sum()
        with torch.autograd.detect_anomaly():
            self.optim.zero_grad()
            score.backward(retain_graph=True)
            self.optim.step()
        

    def forward(self, state):
        x = self.layers(torch.FloatTensor(state))
        return F.softmax(x, dim=-1)

    def get_params(self):
        return self.parameters()
