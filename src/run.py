import matplotlib.pyplot as plt
import gymnasium as gym
import torch

#this needs to go up here bc of a bug
if torch.cuda.is_available():
    torch.cuda.init()

from models.policy import (AbstractPolicy, LinearDiscretePolicy, LinearGaussianPolicy, NNDiscretePolicy)
from agents.policygradient import PolicyGradientAgent
from agents.reinforce import REINFORCEAgent, BatchREINFORCEAgent


if __name__ == '__main__':

    ENV = gym.make('LunarLander-v2', render_mode='rgb_array')

    actor_fn = NNDiscretePolicy(ENV.observation_space, ENV.action_space, hidden_size=16, alpha=1e-3)
    # actor_fn = NNGaussianPolicy(ENV.observation_space, ENV.action_space, alpha=1e-2)
    # actor_fn = LinearDiscretePolicy(ENV.observation_space, ENV.action_space, alpha=1e-2)
    # actor_fn = LinearGaussianPolicy(ENV.observation_space, ENV.action_space, alpha=1e-2)

    agent = BatchREINFORCEAgent(
                policy_fn=actor_fn,
                env=ENV,
                gamma=0.98,
                )

    n_epochs = 8 
    for e in range(n_epochs):
        agent.train(100)
        plt.show()
        ENV.render()

