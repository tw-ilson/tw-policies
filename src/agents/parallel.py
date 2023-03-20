import multiprocessing
from copy import deepcopy
from gymnasium import Env

from agents.actorcritic import ActorCriticAgent
from models.policy import AbstractPolicy
from models.returns import QValueNetworkReturns, ValueNetworkReturns
from agents.pgagent import PolicyGradientAgent
from agents.actorcritic import ActorCriticAgent
from src.models.policy.discrete import NNDiscretePolicy

class A2CAgent(PolicyGradientAgent):
    """Advantage Actor Critic is a policy gradient agent with a special focus on parallel training. The critics learn the value function, while mutiple actors are trained in parallel and get synced with global parameters periodically.
    """
    def __init__(self, env: Env, actor_fn: NNDiscretePolicy, critic_fn: QValueNetworkReturns, N, gamma: float = 0.98, ):
        super().__init__(env, actor_fn, critic_fn, gamma )
        agent = ActorCriticAgent(env, self.policy, self.returns)
        self.children = [deepcopy(agent) for _ in range(N)]

    def train(self, n_iter):
        return super().train(n_iter)

