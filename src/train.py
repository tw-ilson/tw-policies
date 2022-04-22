import numpy as np
from matplotlib import pyplot as plt
import torch

import gym
from policy_functions import LinearDiscretePolicyApproximator, NNDiscretePolicyApproximator, NNGaussianPolicyApproximator, LinearGaussianPolicyApproximator
from learner import REINFORCEAgent, VanillaPolicyGradientAgent

if __name__ == '__main__':
    # env = gym.make("ALE/Pong")
    # env = gym.make("Taxi-v3")
    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v2")
    # env = gym.make("LunarLanderContinuous-v2")

    # nn_pi = NNDiscretePolicyApproximator(env.observation_space, env.action_space, alpha=1e-3)
    l_pi = LinearDiscretePolicyApproximator(env.observation_space, env.action_space, alpha=1e-3)
    # gauss_pi = LinearGaussianPolicyApproximator(env.observation_space, env.action_space, alpha=1e-4)

    agent = REINFORCEAgent(
                policy_fn=l_pi,
                gamma=0.99,
                env=env)

    # agent = VanillaPolicyGradientAgent(
    #             policy_fn=torch.load('LunarLander-v2.pt'),
    #             gamma=0.98,
    #             env=env,
    #             batch_size=2)

    agent.train(5000)
    plt.show()
    
    # torch.save(agent.pi, f'{env.unwrapped.spec.id}.pt')

    while 1:
        agent.playout(render=True)

    exit()
