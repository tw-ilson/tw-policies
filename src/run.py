import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from agents.actorcritic import ActorCriticAgent

from models.returns.valuenetwork import QValueNetworkReturns

#this needs to go up here
if torch.cuda.is_available():
    torch.cuda.init()

from models.policy import (AbstractPolicy, LinearDiscretePolicy, LinearGaussianPolicy, NNDiscretePolicy)
from agents.pgagent import PolicyGradientAgent
from agents.reinforce import REINFORCEAgent

'''
This file provides a simple example usage of the library and its usage. As you can see, all that is needed to run training, given a Gym environment, is the initialization of the policy (an instance of AbstractPolicy) and the initialization of the Agent (instance of PolicyGradientAgent). Note that the design choice to separate these things allows for experimentation with policies within the same iteration framework.
'''


if __name__ == '__main__':

    ENV = gym.make('CartPole-v1')

    # policy = LinearDiscretePolicy(ENV.observation_space, ENV.action_space, alpha=1e-3)
    policy = NNDiscretePolicy(ENV.observation_space, ENV.action_space, alpha=1e-3)
    q_network = QValueNetworkReturns(ENV.observation_space, ENV.action_space, hidden_size=16, n_hidden=0)

    agent = ActorCriticAgent(
                env=ENV,
                actor_fn=policy,
                critic_fn=q_network
                )

    n_epochs = 4 
    for e in range(n_epochs):
        agent.train(10000)
        plt.show()
        agent.playout()

