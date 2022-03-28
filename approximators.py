from typing import List

import abc
import math
import torch
import numpy as np
import torch.nn as nn
from torch.optim import SGD

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

    @abc.abstractmethod
    def __init__(self, state_dim, action_space, alpha) -> None:
        self.state_dim = state_dim.shape
        self.action_space = action_space.n
        self.alpha = alpha

    @abc.abstractmethod
    def get_params(self):
        pass

    @abc.abstractmethod
    def compute_score(self, s_t, a_t, A):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; performs backwards pass to updates network parameters according to score calculation.
        Params
        ------
            s_t: state Tensor used as input to this policy (B, *state_shape)
            a_t: vector mask of action taken (B)
            A: advantage multiplier (B)
        Returns
        ------
            score: gradient of log policy probability weighted by advantage
    '''


    @abc.abstractmethod
    def __call__(self, state:np.ndarray) -> np.ndarray:
        '''Given a state, return an approximation of the log probability vector of the action space distribution
        '''
        pass

        

class MLP(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.sequence = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(True),
                nn.Linear(128, 128),
                nn.ReLU(True),
                nn.Linear(128, 128),
                nn.ReLU(True),
                nn.Linear(128, 128),
                nn.ReLU(True),
                nn.Linear(128, output_size),
            )

    def forward(self, x):
        return self.sequence(x)


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
        # print(self.state_dim)
        # print(self.action_space)

        #initialize weights matrix, single fully connected layer, no bias because meh
        self.weights = np.random.uniform(0, 1, size=(self.action_space, *self.state_dim))


        # 'Surrogate' Loss function
        # self.loss = lambda y: \
        #         np.sum(np.log(y))

    def get_params(self):
        return self.weights
    
    def compute_score(self, s_t:np.ndarray, a_t:np.ndarray, A):
        ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
            which follows a single linear combination layer of the state input.
        '''

        sm_out = self.__call__(s_t)

        #one-hot encode action
        act_mask = np.zeros(self.action_space)
        act_mask[a_t] = 1 * A

        #The components of the gradient via chain rule
        dL_dsm = np.reciprocal(sm_out) + 1e-9
        SM =  sm_out.reshape((-1,1))
        Jsm_dx = np.diagflat(sm_out) - np.dot(SM, SM.T)
        dx_dw = s_t

        #gradient descent step (no jax!)
        d_weights = np.outer(dx_dw, dL_dsm * np.dot(Jsm_dx, act_mask))

        self.weights = self.weights - self.alpha * d_weights.T

    def __call__(self, state: np.ndarray) -> np.ndarray:
        #SoftMax following output layer
        soft_max = lambda x: \
                np.exp(x)/sum(np.exp(x))
        y = np.log(soft_max(np.dot(self.weights, state) ))
        return y

class NNPolicyApproximator(AbstractPolicyApproximator, nn.Module):
    '''The full policy network composed of convolutional and MLP parts '''

    def __init__(self,
            state_space:Space,
            action_space:Space,
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicyApproximator.__init__(self, state_space, action_space, alpha)

        self.layers = []
        if not from_image:
            assert self.state_dim == 1
            self.layers.append(MLP( \
                    state_space.shape, \
                    action_space.shape))
        else:
            # self.layers.append(CNN())
            raise NotImplemented
        
        self.output_layer = nn.Softmax()
        self.layers.append(self.output_layer)
        self.layers = nn.Sequential(self.layers)

        self.optim = SGD(self.get_params(), lr=self.apha)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.forward(state)
        
    def compute_score(self, s_t, a_t, A):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; updates network parameters according to score calculation.
        Params
        ------
            s_t: state Tensor used as input to this policy (B, *state_shape)
            a_t: index of action taken (B)
            A: advantage multiplier (B)
        Returns
        ------
            score: gradient of log policy probability weighted by advantage
        '''

        # get policy distribution of action (log probabilities)
        a_dist = self.forward(s_t)
        a_mask = torch.zeros(a_dist.shape)
        a_mask[a_t] = A
        loss = torch.sum(a_dist * a_mask, dim=-1)
        a_mask.backward
        loss.backward()
        return loss

    def forward(self, state):
        return self.output_layer(self.layers(state))

    def get_params(self):
        return self.parameters()
