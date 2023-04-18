import torch
from torch.nn.functional import kl_div
import random
from copy import deepcopy

from ..models.value import GAEReturns
from . import PolicyGradientAgent
from ..utils import prepare_batch

class PPOAgent(PolicyGradientAgent):
    def __init__(self, env, policy, sigma_target, gamma: float = 0.98) -> None:
        super().__init__(env, policy, gamma)
        self.returns = GAEReturns(state_space=env.observation_space, lam=0.92, gamma=gamma)

        self.oldpolicy = deepcopy(self.policy)
        self.sigma_target = sigma_target # Target KL divergence for training
        self.beta = random.random() # tolerance

        self.plot_info['importance'] = []
        self.plot_info['KL div'] = []

    def update(self, episode):
        tau = self.playout()
        states, actions, rewards, dones = tau
        A = self.returns(states, rewards)
        states, actions, A = prepare_batch(states[:-1], actions, A)
        # Trust Region constraint
        pdf = self.policy.pdf(states)
        old_pdf = self.oldpolicy.pdf(states)
        importance = pdf.log_prob(actions)/old_pdf.log_prob(actions) # importance sampling 
        sigma = kl_div(pdf.log_prob(actions), old_pdf.log_prob(actions), log_target=True) 
        # Loss = importance sampling 
        loss = (importance*A).mean() - self.beta * sigma
        self.oldpolicy = deepcopy(self.policy)
        # self.oldpolicy.load_state_dict(self.policy.state_dict())
        self.policy.optimize(-loss)
        self.returns.optimize(states, rewards)

        if sigma < self.sigma_target/1.5:
            self.beta /= 2
        elif sigma > self.sigma_target * 1.5:
            self.beta *= 2
        self.plot_info['importance'].append(importance.sum())
        self.plot_info['KL div'].append(sigma)
