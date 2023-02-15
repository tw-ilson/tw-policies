from typing import ( TYPE_CHECKING, List, Tuple, Callable, Any )
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
# import ale_py
import gymnasium as gym
from tqdm import tqdm
from abc import ABC, abstractmethod

from utils import plot_curves
from models.policy import AbstractPolicy
from models.reward import AbstractReturns, MonteCarloReturns 

class PolicyGradientAgent(ABC):
    ''' Abstract class defining general properties and utilities for policy gradient reinforcement learning algorithms
    
        Attributes
        ----------
            gamma: discount factor for learning delayed rewards
            alpha: learning rate used for gradient optimization of weights
            epsilon: randomness factor used for exp
            policy: function with which to choose actions based on current state
            baseline: baseline function with which to compute error from return based on state
            env: world environment in which our agent's decision process is formulated
        
    '''

    def __init__(self,
            gamma:float=0.98,
            env=gym.Env) -> None:

        self.env = env  
        self.gamma = gamma

        self.start_obs, info = self.env.reset()
        self.last_obs = self.start_obs

        #define policy function and advantage
        self.pi = ...
        self.value = ...

        self.n_iter = ...

        self.plot_info = {
                'playout advantage': [],
                'score': []
                }

    def playout(self, n_steps:int=0, render:bool=False):
        '''Collects a full playout, or a number of discrete timesteps, and collects the transitions in three arrays
            Note: There should be no more than 1 call to policy.optimize per call to this function
        '''
        step_i = 0
        states = []
        actions = []
        rewards = []

        if n_steps > 0:
            assert not isinstance(self.value, MonteCarloReturns), 'MonteCarloReturns requires full playouts'
            loop_pred = lambda: step_i < n_steps
        else:
            # assert bytes(self.last_obs) == bytes(self.start_obs), 'episode ended unexpectedly?'
            loop_pred = lambda: not done

        s_t = self.last_obs
        done = False
        #Do playout
        while loop_pred():
            step_i += 1
            s_prev = s_t
            a_t = self.pi(s_t)
            s_t, reward, done, trun, info = self.env.step(a_t)
            states.append(s_prev)
            actions.append(a_t)
            rewards.append(reward)
            # if render:
            #     self.env.render()
            if done:
                s_t, info = self.env.reset()
                self.start_obs = s_t
        self.last_obs = s_t
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32)

    @abstractmethod
    def train(self, n_iter):
        '''This method will implement the policy gradient algorithm for each respective subclass'''
        pass

    def display_plots(self):
        print('plotting...')
        plot_curves(**self.plot_info)


