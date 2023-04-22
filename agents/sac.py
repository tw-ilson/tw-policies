from .pgagent import PolicyGradientAgent

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
