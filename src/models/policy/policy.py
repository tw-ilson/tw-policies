from abc import ABC, abstractmethod
from typing import Tuple

import gymnasium as gym
from gymnasium.spaces.utils import flatdim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from models.cnn import CNN 
from utils import logsumexp, prepare_batch

class AbstractPolicy(ABC):
    '''Abstract class to represent a stochastic policy function approximator.
    '''

    def __init__(self, state_space:gym.Space, action_space:gym.Space, lr:float=0.001, continuous_action:bool=False, deterministic:bool=False) -> None:
        print(f"State space: {state_space}\nAction space: {action_space}")
        assert(state_space.is_np_flattenable)
        assert(action_space.is_np_flattenable)
        self.continuous_action = continuous_action
        self.state_space = state_space
        self.state_dim = flatdim(state_space)
        self.action_space = action_space
        self.action_dim = flatdim(action_space)
        self.lr = lr

    @abstractmethod
    def pdf(self, state):
        '''Returns the log probability density function of this policy distribution
        '''
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def score(self, s, a, v):
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
        '''Given a state, choose an action
        '''
        pass
