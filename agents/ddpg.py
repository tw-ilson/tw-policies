import torch
from . import PolicyGradientAgent
from ..models.policy import DeterministicPolicy
from ..models.value import GAEReturns, QValueNetworkReturns
from ..models.replay  import ReplayBuffer
from copy import deepcopy

class DDPGAgent(PolicyGradientAgent):
    """Deep Deterministic Policy Gradient algorithm
        this off-policy algorithm introduces The replay buffer as well as target policy and target q-network
    """
    def __init__(self,
                 env,
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
