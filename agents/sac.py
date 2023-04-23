from .pgagent import PolicyGradientAgent
from ..models.value import QValueNetworkReturns
from ..models.policy import AbstractPolicy
from ..models.replay import ReplayBuffer
from ..utils import prepare_batch
from copy import deepcopy
import torch
from torch import nn, set_float32_matmul_precision
from torch.nn.functional import mse_loss, kl_div

class SACAgent(PolicyGradientAgent):
    def __init__(self, 
                 env,
                 policy: AbstractPolicy,
                 batch_size = 128,
                 buffer_size = 5000,
                 gamma = 0.98,
                 alpha = 0.01,
                 tau=1e-2) -> None:
        super().__init__(env, policy, gamma)
        self.batch_size = batch_size
        self.tau = tau

        # Clipped double-Q method
        self.Qa = QValueNetworkReturns(self.env.observation_space, self.env.action_space)
        self.Qb = QValueNetworkReturns(self.env.observation_space, self.env.action_space)
        self.Qa_target = deepcopy(self.Qa)
        self.Qb_target = deepcopy(self.Qb)

        # Automatic entropy tuning
        # Target Entropy = −dim(A)
        self.target_entropy = -torch.prod(torch.tensor(self.policy.action_dim)).item()
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.policy.lr)

        # Initialize a replay buffer
        self.replay = ReplayBuffer(
                buffer_size,
                self.policy.state_space.shape,
                self.policy.state_space.shape,
                continuous=True)

        self.plot_info['Q loss'] = []
        self.plot_info['policy loss'] = []
        self.plot_info['alpha loss'] = []

    def soft_target_update(self):
        for target_param, param in zip(self.Qa_target.parameters(), self.Qa.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.Qb_target.parameters(), self.Qb.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def update(self, episode):
        s, a, r, done =  self.step()
        if done:
            sp = self._term
        else:
            sp = self.last_obs
        self.replay.add_transition(s, a, r, sp, done)
        if episode > self.batch_size:
            batch = self.replay.sample(self.batch_size)
            states, actions, rewards, next_states, dones = prepare_batch(*batch)

            # Calculate Q target value, with penalty
            with torch.no_grad():
                next_state_action = self.policy(next_states)
                next_state_log_prob = self.policy.pdf(next_states).log_prob(next_state_action)
                qa_next_target = self.Qa_target(next_states, next_state_action)
                qb_next_target = self.Qb_target(next_states, next_state_action)
                #Q-Value target at next state, with entropy penalty
                min_q_next_target = torch.min(qa_next_target, qb_next_target) - self.alpha * next_state_log_prob
                q_target = rewards + (1 - dones) * self.gamma * min_q_next_target

            # optimize Q networks
            qa = self.Qa(states, actions)
            qb = self.Qb(states, actions)
            qa_loss = mse_loss(qa, q_target)
            qb_loss = mse_loss(qb, q_target)
            self.Qa.optimize(qa_loss)
            self.Qb.optimize(qb_loss)

            # Sample from policy distribution
            sample_actions = self.policy(states)
            log_prob_pi = self.policy.pdf(states).log_prob(sample_actions)
            qa_pi = self.Qa(states, sample_actions)
            qb_pi = self.Qb(states, sample_actions)
            min_q_pi = torch.min(qa_pi, qb_pi)

            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            policy_loss = ((self.alpha * log_prob_pi) - min_q_pi).mean()
            self.policy.optimize(policy_loss)

            alpha_loss = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

            self.soft_target_update()
