from abc import ABC, abstractmethod
from typing import Tuple
import gym as gym
from gym.spaces.utils import flatdim
import numpy as np

class AbstractPolicy(ABC):
    '''Abstract class to represent a stochastic policy function approximator.
    '''

    def __init__(self, state_space:gym.Space, action_space:gym.Space, lr:float=1e-3, continuous_action:bool=False, deterministic:bool=False) -> None:
        print(f"State space: {state_space}\nAction space: {action_space}")
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
        raise NotImplemented

    @abstractmethod
    def get_params(self):
        raise NotImplemented

    @abstractmethod
    def score(self, s, a, v):
        ''' Given a full batch of transition using this policy, computes the expectation of the score gradient scaled by the advantage
        Params
        ------
            s: states (array-like)
            a: actions (array-like)
            v: value/advantages (array-like)

        '''
        raise NotImplemented

    @abstractmethod
    def optimize(self, score) -> None:
        '''  Performs backwards pass to updates network parameters according to score calculation.
        Params
        ------
            score: the 'score function' as described in PG literature, typically computed using log probability of action scaled by 
            advantage: the advantage of the state or sequence from which the score was computed.
        '''
        raise NotImplemented

    @abstractmethod
    def __call__(self, state:np.ndarray):
        '''Given a state, choose an action
        '''
        raise NotImplemented
