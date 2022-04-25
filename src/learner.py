from typing import List, Tuple, Callable, Any
from collections import OrderedDict
import sys
import os

import numpy as np
from matplotlib import pyplot as plt
import torch

import ale_py

import gym
from tqdm import tqdm
import pickle

from ale_py import ALEInterface, SDL_SUPPORT

ale = ALEInterface()
# ale.loadROM(Pong)
# # Check if we can display the screen
if SDL_SUPPORT:
    ale.setBool("sound", True)
    ale.setBool("display_screen", True)


from policy_functions import NNDiscretePolicy, LinearDiscretePolicy
from returns import AbstractReturns, MonteCarloReturns, MonteCarloBaselineReturns
from utils import ReplayBuffer


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

        #define policy function and advantage
        self.pi = ...
        self.G = ...

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

    # def compute_return_advantage(self, playout):
    #         trajectory = OrderedDict()
    #         t = len(playout) - 1
    #         s_t, a,  reward, log_prob = playout[-1]
    #         trajectory[bytes(s_t)] = reward, reward - self.baseline(s_t)
    #         while t > 0:
    #             t-=1
    #             s_t, a, reward, log_prob = playout[t]
    #             ret_tnext = trajectory[bytes(playout[t+1][0])][0]
    #             ret_t = reward + self.gamma * ret_tnext
    #             adv_t = ret_t - self.baseline(s_t)
    #             trajectory[bytes(s_t)] = ret_t, adv_t
    #
    #         return trajectory

    def train(self, n_iter):
        '''This method will implement the policy gradient algorithm for each respective subclass'''
        pass


class REINFORCEAgent(PolicyGradientAgent):
    ''' REINFORCE is a simple monte-carlo policy gradient algorithm. The algorithm simply samples multiple trajectories following the current policy while updating policy approximation function weights using the estimated gradient. Note that a baseline is not used for the gradient update calculation in this version of the algorithm
    '''

    def __init__(self, 
            policy_fn:Callable,
            gamma: float = 0.99, 
            env: gym.Env = None) -> None:
        super().__init__(gamma, env)
        
        self.pi = policy_fn
        self.G = MonteCarloReturns(self.gamma)


    def train(self, n_iter):
        pbar = tqdm(range(1, n_iter+1), file=sys.stdout)
        episode_rewards = []
        scores = []
        for episode in pbar:
            tau = self.playout(render=True)
            self.G.update_baseline(tau)

            score = self.pi.compute_score(
                        tau,
                        self.G)

            self.pi.update_params(score)

            episode_rewards.append(sum(r for _,_, r, _ in tau))
            scores.append(score)
            pbar.set_description(f'avg. return == {sum(episode_rewards)/episode:.3f}')
        plot_curves(episode_rewards, scores)


class VanillaPolicyGradientAgent(PolicyGradientAgent):
    '''This class is an implementation of the 'vanilla' policy gradient algorithm, which provides a general framework for nearly all other gradient-optimization approaches to policy-based Reinforcement Learning. It is very similar to the REINFORCE algorithm. However, in VPG we are calculating gradients based on a batch of playouts as opposed to a single playout; additionally, the essential difference is that we are adding a BASELINE in order to ensure that better trajectories produce positive scores, while worse trajectories produce negative scores. This helps to smooth the optimization landscape. This introduces a notion of the value of a state into our policy-based method.
    '''

    def __init__(self, 
            policy_fn:Callable, 
            G:AbstractReturns=None, 
            batch_size=16, 
            gamma: float = 0.98, 
            env: gym.Env=gym.make('CartPole-v1')) -> None:

        super().__init__(gamma, env)
        self.batch_size = batch_size

        self.buffer = ReplayBuffer(400, self.pi.state_dim, self.pi.action_space)
        self.pi = policy_fn
        self.G = G
        if G is None:
            self.G = MonteCarloReturns(self.gamma)

    def train(self, n_iter):
        self.n_iter = n_iter
        avg_batch_rewards = []
        scores = []

        pbar = tqdm(range(1, n_iter+1))
        for episode in pbar:
            batch_reward = 0
            batch_loss = 0
            for _ in range(self.batch_size):
                tau = self.playout()
                batch_loss += self.pi.compute_score(tau, self.G)
                batch_reward += sum(r for _,_, r, _ in tau)
            self.G.update_baseline(tau)
            self.pi.update_params(batch_loss)

            avg_batch_rewards.append(batch_reward/self.batch_size)
            scores.append(batch_reward)
            pbar.set_description(f'avg. return == {sum(avg_batch_rewards)/episode:.3f}')

                
def plot_curves(rewards, loss):
    f, axs = plt.subplots(1, 2, figsize=(7,2.5))
    W = 50 # smoothing window

    [a.clear() for a in axs]
    axs[0].plot(np.convolve(rewards, np.ones(W)/W, 'valid'))
    axs[0].set_xlabel('episodes')
    axs[0].set_ylabel('episodic rewards')

    if len(loss) > 0:
        axs[1].plot(np.convolve(loss, np.ones(W)/W, 'valid'))
    axs[1].set_xlabel('opt steps')
    axs[1].set_ylabel('score')
    plt.tight_layout()

if __name__ == '__main__':
    # env = gym.make("ALE/Pong")
    env = gym.make("LunarLander-v2")
    policy_fn = NNDiscretePolicy(env.observation_space, env.action_space, alpha=1e-3, from_image=False)
    # policy_fn = LinearDiscretePolicy(env.observation_space, env.action_space, alpha=1e-3)


    agent = REINFORCEAgent(
                policy_fn=policy_fn,
                gamma=0.98,
                env=env)

    # agent = VanillaPolicyGradientAgent(
    #             policy_fn=policy_fn,
    #             gamma=0.98,
    #             env=env,
    #             batch_size=4)

    n_epochs = 2 
    for e in range(n_epochs):
        agent.train(10000)
        plt.show()
        agent.playout(render=True)

    with open(f'{env.unwrapped.spec.id}.pt', 'xw') as f:
        torch.save(agent.pi, f)

    exit()

