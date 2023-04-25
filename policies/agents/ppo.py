import torch
from torch.nn.functional import kl_div
import random
from copy import deepcopy

from ..models.replay import ReplayBuffer

from ..models.value import GAEReturns
from . import PolicyGradientAgent
from ..utils import prepare_batch
from torchviz import make_dot

class PPOAgent(PolicyGradientAgent):
    def __init__(self, env, policy, eps, gamma: float = 0.98, batch_size=128, buffer_size=5000) -> None:
        super().__init__(env, policy, gamma)
        self.returns = GAEReturns(state_space=env.observation_space, lam=0.92, gamma=gamma)

        self.newpolicy = deepcopy(self.policy)
        self.beta = random.random() # tolerance
        self.eps = eps #clipping

        self.plot_info['importance'] = []
        self.plot_info['entropy'] = []

        self.batch_size = batch_size
        self.replay = ReplayBuffer(
                buffer_size,
                self.policy.state_space.shape,
                self.policy.action_space.shape,
                continuous=self.policy.is_continuous)

    
    def copy_weights_old(self):
        with torch.no_grad():
            self.policy.load_state_dict(self.newpolicy.state_dict())

    def update(self, episode):
        self.copy_weights_old()

        # Add playout to buffer
        states, actions, rewards, dones = self.playout()
        A = self.returns(states, rewards)
        for i in range(1, len(states)-1):
            self.replay.add_transition(states[i-1], actions[i], A[i], states[i], dones[i])

        # optimize V
        self.returns.optimize(states[:-1], rewards)

        if len(self.replay) > self.batch_size:
            batch = self.replay.sample(self.batch_size)
            states, actions, advs, next_states, dones = prepare_batch(*batch)

            states, actions, advs = prepare_batch(states, actions, advs)
            pdf = self.newpolicy.pdf(states)
            with torch.no_grad():
                old_pdf = self.policy.pdf(states)

            # The PPO loss function (subscript t is omitted)
            # L^{CLIP+VF+S} = E[ L^CLIP - c1 * L^VF + c2 * H[pi](s) ]
            log_probs = pdf.log_prob(actions)
            old_log_probs = old_pdf.log_prob(actions)
            importance = torch.exp(log_probs - old_log_probs) # p(a|s)/p_old(a|s)
            loss_clip = -torch.min(importance*advs, torch.clamp(importance, 1-self.eps, 1+self.eps)*advs).mean()
            penalty = -self.beta * pdf.entropy().mean()

            loss = loss_clip + penalty

            self.newpolicy.optimize(loss)

            self.plot_info['importance'].append(importance.sum())
            self.plot_info['entropy'].append(penalty)
