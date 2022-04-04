from typing import Dict, Set, List, Tuple, Callable, Any
from collections import OrderedDict
import sys
import os

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.distributions import Categorical

import gym
import random
from tqdm import tqdm
import pickle

####################################################
# Atari Learning Environment 
import ale_py
from ale_py import ALEInterface, SDL_SUPPORT
from ale_py.roms import Galaxian

# ale = ALEInterface()
# ale.loadROM(Galaxian)
#
# # Check if we can display the screen
# if SDL_SUPPORT:
#     ale.setBool("sound", True)
#     ale.setBool("display_screen", True)
###################################################

from policy_functions import NNPolicyApproximator, LinearPolicyApproximator
from baselines import TDBaseline


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

        self.start_state = ...

        self.pi = ...

        #placeholder baseline is zero
        self.baseline = lambda r: 0

        self.i = 0
        self.n_iter = ...


    def playout(self, render:bool=False):
            t=0 
            current_playout = []

            s_t = self.env.reset()
            self.start_state = s_t
            reward = 0
            done = False

            #Do full playout
            while not done:
                t+=1
                s_prev = s_t
                #choose an action based on policy
                a, log_prob = self.pi(s_t)
                # Apply an action and get the resulting reward
                s_t, reward, done, info = self.env.step(a)

                current_playout.append((s_prev, a, reward, log_prob))
                if render:
                    self.env.render()

            return current_playout

    def compute_return_advantage(self, playout):
            trajectory = OrderedDict()
            t = len(playout) - 1
            s_t, a,  reward, log_prob = playout[-1]
            trajectory[bytes(s_t)] = reward, reward - self.baseline(s_t)
            while t > 0:
                t-=1
                s_t, a, reward, log_prob = playout[t]
                ret_tnext = trajectory[bytes(playout[t+1][0])][0]
                ret_t = reward + self.gamma * ret_tnext
                adv_t = ret_t - self.baseline(s_t)
                trajectory[bytes(s_t)] = ret_t, adv_t

            return trajectory

    def train(self, n_iter):
        '''This method will implement the policy gradient algorithm for each respective subclass'''
        pass


class REINFORCEAgent(PolicyGradientAgent):
    ''' REINFORCE is a simple monte-carlo policy gradient algorithm. The algorithm simply samples multiple trajectories following the current policy while updating policy approximation function weights using the estimated gradient. Note that a baseline is not used for the gradient update calculation in this version of the algorithm
    '''

    def __init__(self, 
            policy_fn:Callable,
            gamma: float = 0.98, 
            env: gym.Env = None) -> None:
        super().__init__(gamma, env)
        
        self.pi = policy_fn


    def train(self, n_iter):
        self.n_iter = n_iter
        pbar = tqdm(range(1, n_iter+1), file=sys.stdout)
        episode_rewards = []
        for episode in pbar:
            scores = []
            total = 0
            tau = self.playout()
            G_tau = self.compute_return_advantage(tau)
            for s_t, a_t, r, log_prob in tau:
                R = G_tau[bytes(s_t)][0]
                score = self.pi.compute_score(s_t, a_t, log_prob, R)
                scores.append(score)

                total += r
            self.pi.update_params(scores)
            episode_rewards.append(total)
            pbar.set_description(f'avg. return == {sum(episode_rewards)/episode}')
        plot_results(episode_rewards)



class VanillaPolicyGradientAgent(PolicyGradientAgent):
    '''This class is an implementation of the 'vanilla' policy gradient algorithm, which provides a general framework for nearly all other gradient-optimization approaches to policy-based Reinforcement Learning. It is very similar to the REINFORCE algorithm. However, in VPG we are calculating gradients based on a batch of playouts as opposed to a single playout; additionally, the essential difference is that we are adding a BASELINE in order to ensure that better trajectories produce positive scores, while worse trajectories produce negative scores. This helps to smooth the optimization landscape. This introduces a notion of the value of a state into our policy-based method.
    '''

    def __init__(self, 
            policy_fn:Callable, 
            baseline:Callable, 
            batch_size=40, 
            gamma: float = 0.98, 
            env: gym.Env=None) -> None:

        super().__init__(gamma, env)
        self.batch_size = batch_size

        self.pi = policy_fn
        self.baseline = baseline

    def train(self, n_iter):
        self.n_iter = n_iter

        for _ in tqdm(range(n_iter)):
            trajectories = list()
            for _ in range(self.batch_size):
                tau = self.playout()
                G_tau = self.compute_return_advantage(tau)
                trajectories.append((tau, G_tau))

            # weights = weights + alpha * G_tau[s_t] * grad_log(policy(a_t | s_t))
            for s_t, a_t, r in tau:
                self.pi.zero_grad()
        

def plot_results(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    policy_fn = NNPolicyApproximator(env.observation_space, env.action_space, alpha=1e-3)
    # policy_fn = LinearPolicyApproximator(env.observation_space, env.action_space, alpha=1e-2)

    # agent = REINFORCEAgent(
    #             policy_fn=policy_fn,
    #             gamma=0.9,
    #             env=env)
    with open("policy.model") as model_file:
        agent = pickle.load(model_file)

    with open("training.txt", 'a') as dump:
        n_epochs = 1
        for e in range(n_epochs):
            line = f'epoch {e}: {agent.train(1000)}'
            # dump.write(line)
            agent.playout(render=True)

    with open("policy.model", 'wb') as model_dump:
        pickle.dump(policy_fn, model_dump)

    exit()

