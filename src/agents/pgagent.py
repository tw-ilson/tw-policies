from typing import Optional
import numpy as np
# import ale_py
import gymnasium as gym
from abc import ABC, abstractmethod

from models.policy import AbstractPolicy
from models.returns import AbstractReturns

from utils import plot_curves

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

    # continuous = bool
    deterministic = bool

    def __init__(self,
                 env,
                 policy,
                 returns,
                 deterministic,
                 gamma:float=0.98,
                 n_iter:int=1000,
            ) -> None:

        self.deterministic=deterministic

        self.env = env  
        self.gamma = gamma

        self.policy = policy
        self.returns = returns

        #because we are doing offline learning
        self.start_obs, info = self.env.reset()
        self.last_obs = self.start_obs

        self.n_iter = n_iter

        self.plot_info = {
                'playout advantage': [],
                'score': [],
                'episode length': []
                }

    def playout(self, n_steps:int=0, render:bool=False):
        '''Collects a full playout, or a number of discrete timesteps, and collects the transitions in three arrays
            Note: There should be no more than 1 call to policy.optimize per call to this function
        '''
        step_i = 0
        states = []
        actions = []
        rewards = []
        dones = []

        if n_steps > 0:
            loop_pred = lambda: step_i < n_steps
        else:
            loop_pred = lambda: not done

        s_t = self.last_obs
        done = False
        while loop_pred():
            step_i += 1
            s_prev = s_t
            a_t = self.policy(s_t) 
            s_t, reward, done, trun, info = self.env.step(a_t)
            if np.ndim(a_t) == 0:
                a_t = [a_t]
            if done:
                s_t, info = self.env.reset()
                self.start_obs = s_t
            states.append(np.asarray(s_prev))
            actions.append(np.asarray(a_t))
            rewards.append(reward)
            dones.append(done)
            # if render:
            #     self.env.render()

        self.last_obs = s_t
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                dones)

    @abstractmethod
    def train(self, n_iter):
        '''This method will implement the policy gradient algorithm for each respective subclass'''
        raise NotImplemented

    def display_plots(self):
        print('plotting...')
        plot_curves(**self.plot_info)


