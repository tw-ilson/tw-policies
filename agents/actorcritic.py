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

class A2CAgent(PolicyGradientAgent):
    def __init__(self, env: Env, actor_fn: AbstractPolicy, gamma: float = 0.98, lam=0.92):
        super().__init__(env, actor_fn, gamma)
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

class DDPGAgent(PolicyGradientAgent):
    """Deep Deterministic Policy Gradient algorithm
        this off-policy algorithm introduces The replay buffer as well as target policy and target q-network
    """
    def __init__(self,
                 env: Env,
                 policy: DeterministicPolicy,
                 batch_size: int = 128, 
                 gamma: float = 0.99,
                 buffer_size: int = 5000,
                 tau = 0.12) -> None:

        super().__init__(
                    env,
                    policy,
                    batch_size)

        self.returns = QValueNetworkReturns(env.observation_space, env.action_space, lr=1e-3, n_hidden=0, hidden_size=128, gamma=0.99)

        # target update ratio
        self.tau = tau

        self.policy_target = deepcopy(self.policy)
        self.returns_target = deepcopy(self.returns)

        self.batch_size = batch_size
        # self.target_update = target_update
        self.buffer_size = buffer_size

        # Initialize a replay buffer
        self.replay = ReplayBuffer(
                buffer_size,
                self.policy.state_space.shape,
                self.policy.action_space.shape,
                continuous=True)

        self.plot_info = {'episode rewards':[],
                          'TD error': [],
                          'policy loss': []
                          }

    def soft_target_update(self):
        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.returns_target.parameters(), self.returns.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def update(self, episode):
        s, a, r, done =  self.step()
        if done:
            sp = self._term
        else:
            sp = self.last_obs
        
        self.replay.add_transition(s, a, r, sp, done)
        if len(self.replay) > self.batch_size:
            # import pdb; pdb.set_trace()
            #sample batch from previous transitions
            batch  = self.replay.sample(self.batch_size)
            states, actions, rewards, next_states, dones = prepare_batch(*batch)
            # Q-network loss
            with torch.no_grad():
                next_actions = self.policy_target.forward(next_states)
                q_target_value = self.returns_target.forward(next_states, next_actions.detach())
            q_value = self.returns.forward(states, actions)
            critic_loss = torch.nn.functional.mse_loss(
                    q_value.squeeze(),
                    (rewards + self.gamma * q_target_value.squeeze())
                    )
            # Policy network loss
            policy_loss = -self.returns.forward(states, self.policy.forward(states)).mean()

            self.policy.optimize(policy_loss)
            self.returns.optimize(critic_loss)

            self.soft_target_update()

            if done:
                self.plot_info['TD error'].append(critic_loss)
                self.plot_info['policy loss'].append(policy_loss)

# class SACAgent(PolicyGradientAgent):
#     def __init__(self, 
#                  env,
#                  policy: NNGaussianPolicy|DeterministicPolicy, 
#                  batch_size = 128,
#                  buffer_size = 5000,
#                  gamma = 0.98,
#                  alpha = 0.1,
#                  tau=1e-2) -> None:
#         super().__init__(env, policy, gamma)
#         self.batch_size = batch_size
#         self.tau = tau
#         self.alpha = alpha
#
#         # Clipped double-Q method
#         self.Qa = QValueNetworkReturns(self.env.observation_space, self.env.action_space)
#         self.Qb = QValueNetworkReturns(self.env.observation_space, self.env.action_space)
#         self.Qa_target = deepcopy(self.Qa)
#         self.Qb_target = deepcopy(self.Qb)
#
#
#         # Initialize a replay buffer
#         self.replay = ReplayBuffer(
#                 buffer_size,
#                 self.policy.state_space.shape,
#                 self.policy.action_space.shape,
#                 continuous=True)
#
#     def train(self, n_iter):
#         from torch.nn.functional import mse_loss, kl_div
#         pbar = tqdm(range(1,n_iter+1))
#
#         s, a, r, d = self.step()
#         episode_reward = r
#         for episode in pbar:
#             sp, ap, rp, dp = self.step()
#             episode_reward += rp
#             if dp:
#                 self.plot_info['episode rewards'].append(episode_reward)
#                 episode_reward = 0
#             self.replay.add_transition(s, a, r, sp)
#             if episode > self.batch_size:
#                 batch = self.replay.sample(self.batch_size)
#                 states, actions, rewards, next_states = prepare_batch(*batch)
#
#                 log_prob_action = self.policy.pdf(states)(actions)
#                 # V_loss = mse_loss(
#                 #         self.V(states),
#                 #         (self.Q(states, actions) - torch.log(self.policy.pdf(states))).mean(dim=1)
#                 #         )
#
#                 Q_loss = TDErr(
#                             self.Q(states, actions),
#                             self.V_target(next_states),
#                             rewards,
#                             self.gamma)
#
#                 Z = ...
#                 #policy_loss = ( self.Q(states, actions) + torch.log(Z(states))).mean()
#
