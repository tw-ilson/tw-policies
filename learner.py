from typing import Dict, Set, List, Tuple, Callable, Any
from collections import OrderedDict

import torch
from torch.optim import SGD

import gym
import numpy as np
import random
from tqdm import tqdm

####################################################
# Atari Learning Environment 
import ale_py
from ale_py import ALEInterface, SDL_SUPPORT
from ale_py.roms import Galaxian

ale = ALEInterface()
ale.loadROM(Galaxian)

# Check if we can display the screen
if SDL_SUPPORT:
    ale.setBool("sound", True)
    ale.setBool("display_screen", True)
###################################################

from networks import PolicyNetwork

def compute_score(model, s_t, a_t, A):
    ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; updates network parameters according to score calculation.
    Params
    ------
        s_t: state Tensor used as input to this policy (B, *state_shape)
        a_t: index of action taken (B)
        A: advantage multiplier (B)
    Returns
    ------
        score: gradient of log policy probability weighted by advantage
    '''

    # get policy distribution of action (log probabilities)
    a_dist = model.forward(s_t)
    a_mask = torch.zeros(a_dist.shape)
    a_mask[a_t] = A
    loss = torch.sum(a_dist * a_mask, dim=-1)
    a_mask.backward
    loss.backward()
    return loss

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
            alpha:float=1e-3,
            epsilon:float=0.9,
            baseline:Callable,
            env:gym.Env) -> None:

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.pi = PolicyNetwork(action_space=env.observation_space)
        self.baseline = baseline
        self.start_state = self.env.reset()

        self.optim = SGD(self.pi.parameters(), lr=self.alpha)

    def collectTrajectories(self, n_playouts) -> Set[Dict[Any, Tuple[float, float]]]:
        ''' Collects a set of trajectories under a given policy. At each timestep in each trajectory,
        computes Return, Advantage estimate.
        Params
        ------
            n: number of trajectories to collect in total
        Returns
        ------
            trajectories: Set of Dicts containing { (state_t | t)  : (Return_t, AdvantageEstimate_t) }
        '''
        trajectories = set()

        for _ in range(n_playouts):
            #finally, add this dictionary to the set
            trajectories.add(self.playout())

        return trajectories

    def playout(self):

            current_playout = []

            s_t = self.env.reset()
            reward = 0
            done = False

            t=0 #Do full playout
            while not done:
                #choose an action based on policy
                a = self.policy(s_t)
                # Apply an action and get the resulting reward
                s_t, reward, done, info = self.env.step(a)
                current_playout.append((s_t, a, reward))
                t+=1

            return current_playout

    def compute_return_advantage(self, playout):
            trajectory = OrderedDict()
            t = len(playout)
            s_t, r_t = playout[-1]
            trajectory[s_t] = (r_t, self.baseline(s_t))
            while t > 0:
                t-=1
                s_t, reward = playout[t]
                ret_tnext = trajectory[playout[t+1]][0]
                ret_t = reward + self.gamma * ret_tnext
                adv_t = ret_t - self.baseline(s_t)

                trajectory[s_t] = ret_t, adv_t
            return trajectory


    def policy(self, state):
        '''Performs e-greedy action selection to get the next action
        '''
        # Get the list of legal actions
        legal_actions = ale.getLegalActionSet()

        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            return max(self.policy_fn(state))

    def train(self, n_iter):
        '''This method will implement the policy gradient algorithm for each respective subclass'''
        raise NotImplemented


class REINFORCEAgent(PolicyGradientAgent):
    ''' REINFORCE is a simple monte-carlo policy gradient algorithm. The algorithm simply samples multiple trajectories following the current policy while updating policy approximation function weights using the estimated gradient. Note that a baseline is not used for the gradient update calculation in this version of the algorithm
    '''

    def train(self, n_iter):
        for _ in tqdm(range(n_iter)):
            tau = self.playout()
            G_tau = self.compute_return_advantage(tau)
            # weights = weights + alpha * G_tau[s_t] * grad_log(policy(a_t | s_t))
            tqdm.write(f'expected return at {n_iter}: {[self.start_state][0]}')
            for s_t, a_t, r in tau:
                self.pi.zero_grad()
                compute_score(self.pi, s_t, a_t, G_tau[s_t][0])


if __name__ == '__main__':
    agent = REINFORCEAgent(policy_fn=PolicyNetwork(),
            baseline=TDBaseline(),
            env= gym.make("CartPole-v1"))

    agent.train(1000)

