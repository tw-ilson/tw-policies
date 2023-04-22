from gym import Env
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch

from torchviz import make_dot

from . import PolicyGradientAgent
from ..models.policy import AbstractPolicy, NNGaussianPolicy, NNDiscretePolicy, DeterministicPolicy, OUNoise
from ..models.value import TDErr,  QValueNetworkReturns, AdvantageNetworkReturns, GAEReturns
from ..models.replay import ReplayBuffer
from ..utils import prepare_batch

class QActorCriticAgent(PolicyGradientAgent):
    '''This class is an implementation of the 'vanilla' policy gradient algorithm, which provides a general framework for nearly all other gradient-optimization approaches to policy-based Reinforcement Learning. It is very similar to the REINFORCE algorithm. However, in VPG we are calculating gradients based on a batch of playouts as opposed to a single playout; additionally, the essential difference is that we are introducing a general notion of a baseline in order to ensure that better trajectories produce positive scores, while worse trajectories produce negative scores. This helps to smooth the optimization landscape. This introduces a notion of the value of a state into our policy-based method.
    '''

    def __init__(self, 
            env:Env,
            actor_fn:AbstractPolicy, 
            critic_fn:QValueNetworkReturns, 
            batch_size:int=16, 
            gamma: float = 0.98, 
            ) -> None:
        '''
        Params
        ------
            actor_fn: policy function 'pi'. represents probability denstity distribution (see policy_functions)
            critic_fn: Value-function aka Returns aka Advantage, can be approximation or monte-carlo playouts
            env: OpenAI Gym Environment
            batch_size: the # of steps per policy optimization
            gamma: discount factor
            target_update: target update frequency (if == 0, no target networks)
            buffer_size: the size of the experienece replay sampling buffer, (if == 0, online algorithm is used)
        '''

        super().__init__(env, actor_fn)
        self.returns = critic_fn
        self.batch_size = batch_size
        self.plot_info['score'] = []
        self.plot_info['td err'] = []

    def update(self, episode):
        states, actions, rewards, dones = self.playout()
        states = states[1:]
        next_states = states[:-1]

        q_value = self.returns(states, actions)
        q_prime = self.returns(next_states, next_actions)
        # Optimize Policy Estimator
        score = self.policy.score(states, actions, q_value)
        self.policy.optimize(score)

        # Optimize Q-Value Estimator
        td_err = TDErr(
            q_value,
            q_prime,
            torch.Tensor(rewards),
            self.gamma)
        self.returns.optimize(states, actions, td_err)

        self.plot_info['score'].append(score)
        self.plot_info['td err'].append(td_err)

