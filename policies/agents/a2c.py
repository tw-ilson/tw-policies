import torch
from . import PolicyGradientAgent
from ..models.policy import AbstractPolicy
from ..models.value import GAEReturns

class A2CAgent(PolicyGradientAgent):
    def __init__(self, env, actor_fn: AbstractPolicy, gamma: float = 0.98, lam=0.92):
        super().__init__(env, actor_fn, gamma)
        assert not actor_fn.is_continuous
        self.returns = GAEReturns(lam=lam, gamma=gamma, state_space=self.env.observation_space)
        self.plot_info['score'] = []
        self.plot_info['advantage'] = []

    def update(self, episode):
        states, actions, rewards, dones = self.playout()
        advantage = self.returns(states, rewards)

        # Optimize Policy Estimator
        score = self.policy.score(states[:-1,:], actions, advantage)
        self.policy.optimize(score)

        # Optimize Advantage Estimator
        self.returns.optimize(states[:-1,:], rewards)

        self.plot_info['score'].append(score.mean().item())
        self.plot_info['advantage'].append(advantage.mean().item())
