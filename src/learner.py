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

# ale = ALEInterface()
# ale.loadROM(Pong)
# # Check if we can display the screen
# if SDL_SUPPORT:
#     ale.setBool("sound", True)
#     ale.setBool("display_screen", True)


from policy_functions import AbstractPolicyApproximator, NNDiscretePolicyApproximator, LinearDiscretePolicyApproximator
from returns import AbstractReturns, MonteCarloReturns, MonteCarloBaselineReturns, TDMonteCarloReturns
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
        self.policy = ...
        self.advantage_fn = ...

        self.n_iter = ...

    def playout(self, render:bool=False) -> List[Tuple[np.ndarray, np.ndarray, float]]:
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
                a = self.policy(s_t)
                # Apply an action and get the resulting reward
                s_t, reward, done, info = self.env.step(a)

                current_playout.append((s_prev, a, reward))
                if render:
                    self.env.render()

            return current_playout

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
        
        self.policy = policy_fn
        self.advantage_fn = MonteCarloReturns(self.gamma)


    def train(self, n_iter):
        pbar = tqdm(range(1, n_iter+1), file=sys.stdout)
        episode_rewards = []
        scores = []
        for episode in pbar:
            tau = self.playout()
            self.advantage_fn.update_baseline([tau])
            ep_rwd = 0
            ep_score = 0
            for s_t, a_t, r in tau:
                score = self.policy.step_score(s_t, a_t)
                self.policy.update_gradient(
                        s_t,
                        score,
                        self.advantage_fn(s_t)
                        )

                ep_rwd += r
                ep_score += score.sum()

            self.policy.optimize()
            episode_rewards.append(ep_rwd)
            scores.append(ep_score)
            if episode > 50:
                pbar.set_description(f'avg. return == {sum(episode_rewards[episode-50:episode])/50:.3f}')
        plot_curves(reward=np.array(episode_rewards), score=np.array(scores))


class VanillaPolicyGradientAgent(PolicyGradientAgent):
    '''This class is an implementation of the 'vanilla' policy gradient algorithm, which provides a general framework for nearly all other gradient-optimization approaches to policy-based Reinforcement Learning. It is very similar to the REINFORCE algorithm. However, in VPG we are calculating gradients based on a batch of playouts as opposed to a single playout; additionally, the essential difference is that we are adding a BASELINE in order to ensure that better trajectories produce positive scores, while worse trajectories produce negative scores. This helps to smooth the optimization landscape. 
    '''

    def __init__(self, 
            policy_fn:AbstractPolicyApproximator,  
            baseline:Callable=None,
            batch_size=16, 
            buffer_size=400,
            gamma: float = 0.98, 
            env: gym.Env=gym.make('CartPole-v1'),
            replay:bool=False) -> None:

        super().__init__(gamma, env)
        self.batch_size = batch_size

        self.policy = policy_fn

        self.advantage_fn = MonteCarloBaselineReturns(self.gamma, baseline)
        if baseline is None:
            self.advantage_fn = TDMonteCarloReturns(self.gamma)

        if replay:
            self.buffer = ReplayBuffer(400, self.policy.state_dim, self.policy.action_dim)

    def train(self, n_iter):
        self.n_iter = n_iter
        avg_batch_rewards = []
        scores = []

        pbar = tqdm(range(1, n_iter+1))
        for episode in pbar:
            pass
            # batch = [self.playout() for m in range(self.batch_size)]
            # score = sum(self.policy.compute_score(tau) for tau in batch)/self.batch_size
            # self.policy.update_params(score, self.advantage_fn)
            # self.advantage_fn.update_baseline(batch)

            # scores.append(score)
        # pbar.set_description(f'avg. return == {sum(avg_batch_rewards)/episode:.3f}')

                
def plot_curves(**curves):
    f, axs = plt.subplots(1, len(curves), figsize=(7,2.5))
    plt.subplots_adjust(wspace=0.3)
    W = 12 # smoothing window

    [a.clear() for a in axs]
    for i, name in enumerate(curves):
        print(name, curves[name].shape)
        if len(curves[name]) > 0:
            axs[i].plot(np.convolve(curves[name], np.ones(W)/W, 'valid'))
        axs[i].set_xlabel('episodes')
        axs[i].set_ylabel(name)


if __name__ == '__main__':
    # env = gym.make("ALE/Pong")
    # env = gym.make("LunarLander-v2")
    # policy_fn = NNDiscretePolicyApproximator(env.observation_space, env.action_space, alpha=1e-3, from_image=False)
    # policy_fn = LinearDiscretePolicyApproximator(env.observation_space, env.action_space, alpha=1e-3)


    # agent = REINFORCEAgent(
    #             policy_fn=policy_fn,
    #             gamma=0.98,
    #             env=env)

    # agent = VanillaPolicyGradientAgent(
    #             policy_fn=policy_fn,
    #             gamma=0.98,
    #             env=env,
    #             batch_size=4)

    # n_epochs = 2 
    # for e in range(n_epochs):
    #     agent.train(5000)
    #     plt.show()
    #     agent.playout(render=True)
    #     torch.save(agent.pi, f'{env.unwrapped.spec.id}.pt')
    exit()
