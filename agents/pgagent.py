from typing import Optional
import numpy as np
# import ale_py
import gym
from gym.wrappers import normalize
from abc import ABC, abstractmethod

from ..models.policy import AbstractPolicy
from ..models.value import AbstractReturns

from ..utils import plot_curves

class PolicyGradientAgent(ABC):
    ''' Abstract class defining general properties and utilities for policy gradient reinforcement learning algorithms
    
        Attributes
        ----------
            gamma: discount factor for learning delayed rewards
            env: world environment in which our agent's decision process is formulated
            policy: function with which to choose actions based on current state
            value: (optional) value estimator function for more complex action selections
        
    '''
    gamma: float
    env: gym.Env
    policy: AbstractPolicy
    value: Optional[AbstractReturns]

    def __init__(self, env, policy, gamma:float=0.98) -> None:

        self.env = env  
        self.gamma = gamma

        self.policy = policy

        self.last_obs = self.env.reset()
        self.plot_info = {
                'playout advantage': [],
                'score': [],
                'episode length': []
                }

    def step(self):
        """ Select action and take action """
        a_t = self.policy(self.last_obs) 
        obs = self.last_obs
        new_obs, reward, done, info = self.env.step(a_t)
        if done:
            self.last_obs = self.env.reset()
        else:
            self.last_obs = new_obs

        return obs, a_t, reward, done

    def playout(self, render=False):
        done = False
        states = list()
        actions = list()
        rewards = list()
        dones = list()
        while not done:
            s_t, a_t, r_t, done = self.step()
            states.append(s_t)
            actions.append(a_t)
            rewards.append(r_t)
            dones.append(done)
            if render:
                self.env.render()
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        return states, actions, rewards, dones

    @abstractmethod
    def train(self, n_iter):
        '''This method will implement the policy gradient algorithm for each respective subclass'''
        raise NotImplemented

    def display_plots(self):
        print('plotting...')
        plot_curves(**self.plot_info)


