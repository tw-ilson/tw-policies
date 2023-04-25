
# import gymnasium as gym
import gym
import torch
# import minari

if torch.cuda.is_available():
    torch.cuda.init()

import policies.agents as agents
import policies.models.policy as policy
import policies.models.value as value

import pybullet as pb
import pybullet_envs
import pickle

'''
This file provides a simple example usage of the library and its usage. As you can see, all that is needed to run training, given a Gym environment, is the initialization of the policy (an instance of AbstractPolicy) and the initialization of the Agent (instance of PolicyGradientAgent). Note that the design choice to separate these things allows for experimentation with policies within the same iteration framework.
'''

def main():
    # env_name = 'AntBulletEnv-v0'
    # env_name = 'BipedalWalker-v3'
    env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'LunarLanderContinuous-v2'
    # env_name = 'Pendulum-v1'

    learner = 'A2C'

    agent = pick_learner(learner, env_name)
    try:
        agent.train(int(2.5e6))
        # agent.train(1000)
    finally:
        agent.display_plots(filepath=f'policies/plot/{learner}-{env_name}.jpg')
        torch.save(agent, f"policies/weights/{learner}-{env_name}.pickle")

def pick_learner(learner, envname) -> agents.PolicyGradientAgent:
    ENV = gym.make(envname)

    agent: agents.PolicyGradientAgent = None
    if learner == 'REINFORCE':
        # policy = policy.NNGaussianPolicy(ENV.observation_space, ENV.action_space, lr=3e-4, n_hidden=1, hidden_size=128)
        p = policy.NNDiscretePolicy(ENV.observation_space, ENV.action_space, lr=3e-4, n_hidden=1, hidden_size=128)
        agent = agents.ReinforceAgent(
                    ENV,
                    policy,
                    )
    elif learner == 'A2C':
        # policy = policy.NNGaussianPolicy(ENV.observation_space, ENV.action_space, lr=3e-4, n_hidden=1, hidden_size=128)
        p = policy.NNDiscretePolicy(
                ENV.observation_space,
                ENV.action_space,
                lr=3e-4,
                n_hidden=1,
                hidden_size=64)

        agent = agents.A2CAgent(
                    ENV,
                    p)

    elif learner == 'DDPG':
        p = policy.DeterministicPolicy(
                    ENV.observation_space,
                    ENV.action_space,
                    lr=3e-4,
                    n_hidden=2,
                    hidden_size=128,
                    noise=0.1)
        agent = agents.DDPGAgent(
                    ENV,
                    p,
                    batch_size=256,
                    buffer_size=int(1e5),
                    tau=1e-3,
                    gamma=0.98
                    )
    elif learner == 'PPO':
        p = policy.NNGaussianPolicy(ENV.observation_space, ENV.action_space, lr=1e-5, n_hidden=2, hidden_size=64)
        # policy = policy.NNDiscretePolicy(ENV.observation_space,ENV.action_space,lr=3e-4,n_hidden=1,hidden_size=128)
        agent = agents.PPOAgent(
                    ENV,
                    p,
                    eps=0.1
                    )
    elif learner == 'SAC':
        p = policy.NNGaussianPolicy(ENV.observation_space, ENV.action_space, lr=1e-5, n_hidden=2, hidden_size=64)
        agent = agents.SACAgent(ENV, p)

    return agent

if __name__ == "__main__":
    main()
