from agents.policygradient import PolicyGradientAgent
from models.policy import AbstractPolicy
from typing import Callable
import gymnasium as gym
from tqdm import tqdm
from models.reward import MonteCarloReturns
import sys

class REINFORCEAgent(PolicyGradientAgent):
    ''' REINFORCE is a simple monte-carlo policy gradient algorithm. The algorithm simply samples trajectories from the current policy while updating policy approximation function weights using the estimated gradient. Note that a baseline is not used for the gradient update calculation in this version of the algorithm
    '''

    def __init__(self, 
            policy_fn:AbstractPolicy,
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
