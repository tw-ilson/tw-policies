import multiprocessing
from copy import deepcopy
from gymnasium import Env

from agents.actorcritic import ActorCriticAgent
from models.policy import AbstractPolicy
from models.returns import QValueNetworkReturns, ValueNetworkReturns
from agents.pgagent import PolicyGradientAgent
from agents.actorcritic import ActorCriticAgent

class A2CAgent(PolicyGradientAgent):
    """Advantage Actor Critic is a policy gradient agent with a special focus on parallel training. The critics learn the value function, while mutiple actors are trained in parallel and get synced with global parameters periodically.
    """
    def __init__(self, env: Env, actor_fn: AbstractPolicy, critic_fn: QValueNetworkReturns, batch_size: int = 16, gamma: float = 0.98, target_update: int = 400, buffer_size: int = 5000) -> None:
        super().__init__(env, actor_fn, critic_fn, batch_size, gamma, target_update, buffer_size)
        agent = ActorCriticAgent(
        self.children = deepcopy()

