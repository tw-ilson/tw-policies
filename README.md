## Introduction
tw-policies is a simple library/tool for Reinforcement Learning, with a focus in the policy gradient class of algorithms. Notably, it contains numpy-only implementations of the policy gradient for discrete and guassian linear function approximations. It contains several premade algorithm implementations using this framework, such as the REINFORCE algorithm and several Actor-Critic algorithms like A2C and PPO. 

## Usage
You can find the networks and inference code for the policies and state-value evaluation code in the two main modules '''models.policy''' which contains the most essential component: the function approximations for different kinds of policies. and '''models.value'''. You can find premade algorithm implementations in the '''agents'''. Some feed-forward neural networks are available in '''networks'''. 

## Todo:
 - [ ] test PPO implementation
 - [ ] Support offline training
 - [ ] Multi-Headed network outputs
 - [ ] Goal-conditioned training
 - [ ] Shared weights feature extraction for Actor-Critic
 - [ ] automated tuning?
