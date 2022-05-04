from typing import List, Tuple, Callable, Any
from collections import OrderedDict
import sys
import os

import numpy as np
from matplotlib import pyplot as plt
import torch

# import ale_py

import gym
from tqdm import tqdm
import pickle

# from ale_py import ALEInterface, SDL_SUPPORT


# ale = ALEInterface()
# ale.loadROM(Pong)
# # Check if we can display the screen
# if SDL_SUPPORT:
#     ale.setBool("sound", True)
#     ale.setBool("display_screen", True)
#

from policy_functions import *
from value_functions import *
from utils import *


class PolicyGradientAgent():
    ''' Abstract class defining general properties and utilities for policy gradient reinforcement learning algorithms
    
        Attributes
        ----------
            gamma: discount factor for learning delayed rewards
            alpha: learning rate used for gradient optimization of weights
            epsilon: randomness factor used for exploration (not always necessary)
            policy: function with which to choose actions based on current state
            baseline: baseline function with which to compute error from return based on state
            env: world environment in which our agent's decision process is formulated
        
    '''

    def __init__(self,
            gamma:float=0.98,
            env:gym.Env=None) -> None:

        self.env = env  
        self.gamma = gamma

        self.start_obs = self.env.reset()
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
            assert bytes(self.last_obs) == bytes(self.start_obs), 'episode ended unexpectedly?'
            loop_pred = lambda: not done

        s_t = self.last_obs
        done = False
        #Do playout
        while loop_pred():
            step_i += 1
            s_prev = s_t
            a_t = self.pi(s_t)
            s_t, reward, done, info = self.env.step(a_t)
            states.append(s_prev)
            actions.append(a_t)
            rewards.append(reward)
            if render:
                self.env.render()
            if done:
                s_t = self.env.reset()
                self.start_obs = s_t
        self.last_obs = s_t
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32)

    def train(self, n_iter):
        '''This method will implement the policy gradient algorithm for each respective subclass'''
        pass

    def display_plots(self):
        print('plotting...')
        plot_curves(**self.plot_info)


class REINFORCEAgent(PolicyGradientAgent):
    ''' REINFORCE is a simple monte-carlo policy gradient algorithm. The algorithm simply samples multiple trajectories following the current policy while updating policy approximation function weights using the estimated gradient. Note that a baseline is not used for the gradient update calculation in this version of the algorithm
    '''

    def __init__(self, 
            policy_fn:Callable,
            env: gym.Env = None,
            gamma: float = 0.99, 
            ) -> None:
        super().__init__(gamma, env)
        
        self.pi = policy_fn
        self.value = MonteCarloReturns(self.gamma, normalize=False) 

    def train(self, n_iter):
        pbar = tqdm(range(1, n_iter+1), file=sys.stdout)
        for episode in pbar:
            tau = self.playout()
            states, actions, rewards = tau
            self.value.update_baseline(tau)
            for s_t, a_t in zip(states, actions):
                score = self.pi.score(
                        [s_t],
                        [a_t],
                        [self.value(s_t)])
                self.pi.optimize(score)

            self.plot_info['playout advantage'].append(sum(rewards))

            tmp = min(500, len(self.plot_info['playout advantage']))
            avg_return = sum(self.plot_info['playout advantage'][-tmp:-1])/tmp
            pbar.set_description(f"avg. return == {avg_return}")
        self.display_plots()

class BatchREINFORCEAgent(PolicyGradientAgent):
    ''' REINFORCE Algorithm, except we are performing the gradient update with respect to the 'surrogate loss' or mean-score over the entire playout, rather than optimizing at each step. This significantly affects convergence properties.
    '''

    def __init__(self, 
            policy_fn:Callable,
            env: gym.Env = None,
            gamma: float = 0.99, 
            ) -> None:
        super().__init__(gamma, env)
        
        self.pi = policy_fn
        self.value = MonteCarloReturns(self.gamma, normalize=True) #normalizing trick (subtract mean). reduces variance

    def train(self, n_iter):

        pbar = tqdm(range(1, n_iter+1), file=sys.stdout)
        for episode in pbar:
            tau = self.playout()
            states, actions, rewards = tau

            self.value.update_baseline(tau)
            advantages = [self.value(s) for s in states]
            score = self.pi.score(states, actions, advantages)
            self.pi.optimize(score)

            # self.plot_info['score'].append(sum(score))
            self.plot_info['playout advantage'].append(advantages[0])

            tmp = min(500, len(self.plot_info['playout advantage']))
            avg_return = sum(self.plot_info['playout advantage'][-tmp:-1])/tmp
            pbar.set_description(f"avg. return == {avg_return}")
        self.display_plots()

class ActorCriticAgent(PolicyGradientAgent):
    '''This class is an implementation of the 'vanilla' policy gradient algorithm, which provides a general framework for nearly all other gradient-optimization approaches to policy-based Reinforcement Learning. It is very similar to the REINFORCE algorithm. However, in VPG we are calculating gradients based on a batch of playouts as opposed to a single playout; additionally, the essential difference is that we are introducing a general notion of a baseline in order to ensure that better trajectories produce positive scores, while worse trajectories produce negative scores. This helps to smooth the optimization landscape. This introduces a notion of the value of a state into our policy-based method.
    '''

    def __init__(self, 
            actor_fn:AbstractPolicy, 
            critic_fn:AbstractReturns, 
            env: gym.Env=gym.make('CartPole-v1'),
            batch_size:int=16, 
            gamma: float = 0.98, 
            target_update:int=400,
            buffer_size:int=0,
            ) -> None:
        '''
        Params
        ------
            actor_fn: policy function 'pi'. represents probability denstity distribution (see policy_functions)
            critic_fn: Value-function aka Returns aka Advantage, can be approximation or monte-carlo playouts
            env: OpenAI Gym Environment
            batch_size: the # of steps per policy optimization
            gamma: discount factor
            target_update: target update frequency
            buffer_size: the size of the experienece replay sampling buffer, if 0, most recent steps are always used
        '''


        super().__init__(gamma, env)
        self.batch_size = batch_size

        self.pi = actor_fn
        self.value = critic_fn
        
        self.target_pi

        self.buffered = buffer_size > 0
        if self.buffered:
            self.buffer = ReplayBuffer(buffer_size, self.pi.state_space.shape, self.pi.action_space.shape)

    def train(self, n_iter):
        self.n_iter = n_iter
        avg_batch_rewards = []
        scores = []

        pbar = tqdm(range(1, n_iter+1))
        for episode in pbar:
            pass

if __name__ == '__main__':
    # env = gym.make("ALE/Pong")
    # env = gym.make("LunarLander-v2")
    # env = gym.make('CartPole-v1')
    env = gym.make('Pendulum-v1')
    # env = gym.make('CarRacing-v1')
    # env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('MountainCarContinuous-v0')

    # actor_fn = NNDiscretePolicy(env.observation_space, env.action_space, hidden_size=16, alpha=1e-3)
    actor_fn = NNGaussianPolicy(env.observation_space, env.action_space, alpha=1e-2)
    # actor_fn = LinearDiscretePolicy(env.observation_space, env.action_space, alpha=1e-2)
    # actor_fn = LinearGaussianPolicy(env.observation_space, env.action_space, alpha=1e-2)

    agent = BatchREINFORCEAgent(
                policy_fn=actor_fn,
                env=env,
                gamma=0.98,
                )

    n_epochs = 8 
    for e in range(n_epochs):
        agent.train(800)
        plt.show()
        agent.playout(render=True)
    

    exit()

