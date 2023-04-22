from typing import Optional
import numpy as np
import gym
from gym.wrappers import normalize
from abc import ABC, abstractmethod
from tqdm import tqdm

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

        self.episode_rewards = 0
        self.last_obs = np.zeros(self.policy.action_dim)
        self.plot_info = {'episode rewards': []}
        self.reset()

        self._term = np.zeros(self.policy.action_dim) 

    def reset(self):
        self.plot_info['episode rewards'].append(self.episode_rewards)
        self.episode_rewards = 0
        self.last_obs = self.env.reset()

    def step(self):
        """ Select action and take action """
        a_t = self.policy(self.last_obs) 
        obs = self.last_obs
        new_obs, reward, done, info = self.env.step(a_t)
        self.last_obs = new_obs
        if done:
            self._term = new_obs
            self.reset()
        self.episode_rewards += reward
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
        states.append(self._term)
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        return states, actions, rewards, dones

    def train(self, n_iter, smoothing=50):
        '''This method will implement the policy gradient algorithm for each respective subclass'''
        pbar = tqdm(range(1, n_iter+1))
        for episode in pbar:
            self.update(episode)
            print_window = min(smoothing, len(self.plot_info['episode rewards']))
            if print_window > 0:
                smoothed_reward = sum(self.plot_info['episode rewards'][-print_window:])/print_window
                pbar.set_description(f"reward: {smoothed_reward:.3f}")
        pbar.close()

    @abstractmethod
    def update(self, episode):
        """ Implement your algorithm in this method to be called each iteration """
        raise NotImplemented

    def display_plots(self, filepath=None):
        print('plotting...')
        plot_curves(filepath=filepath, **self.plot_info)
