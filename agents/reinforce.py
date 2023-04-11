from typing import Callable
import gym
from tqdm import tqdm
import sys
import torch

from ..models.value.montecarlo import MonteCarloReturns

from . import PolicyGradientAgent
from ..models.policy import AbstractPolicy

class REINFORCEAgent(PolicyGradientAgent):
    ''' REINFORCE is a simple monte-carlo policy gradient algorithm. The algorithm simply samples trajectories from the current policy while updating policy approximation function weights using the estimated gradient. Note that a baseline (normalization technique) is not used for the gradient update calculation in this version of the algorithm. By definition, this algorithm calculates advantages using simple reward-to-go of monte-carlo playouts. We are performing the gradient update with respect to the 'surrogate loss' or mean-score over the entire playout, rather than optimizing at each step. This significantly affects convergence properties, and facilitates the use of the markovian assumption, at the cost of possibly decreasing sample efficiency. Technically that makes this a modified 'batched' REINFORCE algorithm to suppport deeper 

    '''
    def __init__(self, env, policy, gamma: float = 0.98) -> None:
        super().__init__(env, policy, gamma)
        self.returns = MonteCarloReturns(gamma, normalize=True)
        self.plot_info['policy score'] = []

    def update(self, episode):
        tau = self.playout()
        states, actions, rewards, dones = tau
        R = self.returns.reward_to_go(rewards)
        score = self.policy.score(states[:-1], actions, R);
        self.policy.optimize(score) 

        self.plot_info['policy score'].append(score)
        
