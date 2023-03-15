from agents.pgagent import PolicyGradientAgent
from models.policy import AbstractPolicy
from typing import Callable
import gymnasium as gym
from tqdm import tqdm
import sys

from models.policy import policy
from models.returns.montecarlo import MonteCarloReturns

class REINFORCEAgent(PolicyGradientAgent):
    ''' REINFORCE is a simple monte-carlo policy gradient algorithm. The algorithm simply samples trajectories from the current policy while updating policy approximation function weights using the estimated gradient. Note that a baseline (normalization technique) is not used for the gradient update calculation in this version of the algorithm. By definition, this algorithm calculates advantages using simple reward-to-go of monte-carlo playouts. We are performing the gradient update with respect to the 'surrogate loss' or mean-score over the entire playout, rather than optimizing at each step. This significantly affects convergence properties, and facilitates the use of the markovian assumption, at the cost of possibly decreasing sample efficiency. Technically that makes this a modified 'batched' REINFORCE algorithm to suppport deeper 

    '''
        
    def train(self, n_iter):
        pbar = tqdm(range(1, n_iter+1), file=sys.stdout)
        for episode in pbar:
            tau = self.playout()
            states, actions, rewards, dones = tau
            advantages = self.returns(rewards)
            score = self.policy.score(states, actions, advantages);
            self.policy.optimize(score) 

            self.plot_info['playout advantage'].append(sum(rewards))

            tmp = min(500, len(self.plot_info['playout advantage']))
            avg_return = sum(self.plot_info['playout advantage'][-tmp:-1])/tmp
            pbar.set_description(f"avg. return == {avg_return}")
        self.display_plots()

