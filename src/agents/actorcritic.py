from agents.pgagent import PolicyGradientAgent
from models.policy import AbstractPolicy
from models.returns import AbstractReturns
from models.replay import ReplayBuffer
from gymnasium import Env
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from models.returns.valuenetwork import QValueNetworkReturns

class ActorCriticAgent(PolicyGradientAgent):
    '''This class is an implementation of the 'vanilla' policy gradient algorithm, which provides a general framework for nearly all other gradient-optimization approaches to policy-based Reinforcement Learning. It is very similar to the REINFORCE algorithm. However, in VPG we are calculating gradients based on a batch of playouts as opposed to a single playout; additionally, the essential difference is that we are introducing a general notion of a baseline in order to ensure that better trajectories produce positive scores, while worse trajectories produce negative scores. This helps to smooth the optimization landscape. This introduces a notion of the value of a state into our policy-based method.
    '''

    def __init__(self, 
            env:Env,
            actor_fn:AbstractPolicy, 
            critic_fn:QValueNetworkReturns, 
            batch_size:int=16, 
            gamma: float = 0.98, 
            target_update:int=400,
            buffer_size:int=5000,
            ) -> None:
        '''
        Params
        ------
            actor_fn: policy function 'pi'. represents probability denstity distribution (see policy_functions)
            critic_fn: Value-function aka Returns aka Advantage, can be approximation or monte-carlo playouts
            env: OpenAI Gym Environment
            batch_size: the # of steps per policy optimization
            gamma: discount factor
            target_update: target update frequency
            buffer_size: the size of the experienece replay sampling buffer, if 0, most recent steps are always used
        '''


        super().__init__(env, actor_fn, critic_fn)
        self.batch_size = batch_size
        plot_info = {
                'score' : [],
                'td_err' : []
                }
        

    def train(self, n_iter):
        pbar = tqdm(range(1, n_iter+1))

        #sample first action 
        s, a, r, d = self.playout(n_steps=1)
        for episode in pbar:
            sp, ap, rp, dp = self.playout(n_steps=1)

            q_value = self.returns(s, a)
            # Optimize Policy Estimator
            score = self.policy.score(s, a, q_value)
            self.policy.optimize(score)

            # Optimize Q-Value Estimator
            td_err = self.returns.td_err(s, a, r, sp, ap)
            self.returns.optimize(s, a, td_err)
             
            s, a, r, d = sp, ap, rp, dp

            # self.plot_info['playout advantage'].append(sum(rewards))
            # tmp = min(500, len(self.plot_info['playout advantage']))
            # avg_return = sum(self.plot_info['playout advantage'][-tmp:-1])/tmp
            # pbar.set_description(f"avg. return == {avg_return}")
        self.display_plots()

class DDPGAgent(ActorCriticAgent):
    """Deep Deterministic Policy Gradient algorithm
        this off-policy algorithm introduces The replay buffer as well as target policy and target q-network
    """
    def __init__(self,
                 env: Env,
                 actor_fn: AbstractPolicy,
                 critic_fn: AbstractReturns,
                 batch_size: int = 16,
                 gamma: float = 0.98,
                 target_update: int = 400,
                 buffer_size: int = 5000) -> None:

        super().__init__(env, actor_fn, critic_fn, batch_size, gamma, target_update, buffer_size)

        self.target_policy = deepcopy(self.policy)
        self.target_returns = deepcopy(self.returns)
        self.target_update = target_update

        self.replay = ReplayBuffer(
                buffer_size,
                self.policy.state_space.shape,
                self.policy.action_space.shape)

    def train(self, n_iter):
        pbar = tqdm(range(1, n_iter+1))
        for episode in pbar:
            tau = self.playout()
            states, actions, rewards, done = tau

            for i in range(1, len(states)):
                s = states[i-1]
                sp, a, r, d = states[i], actions[i], rewards[i], dones[i]
                self.replay.add_transition(s, a, r, sp, d)
                if self.replay.size > self.batch_size:
                    s, a, r, sp, d = self.replay.sample(self.batch_size)
                    # Optimize Policy Estimator
                    score = self.policy.score(s, a, q_value)
                    self.policy.optimize(score)

                    # Optimize Q-Value Estimator
                    td_err = self.returns.td_err(s, a, r, sp, ap)
                    self.returns.optimize(s, a, td_err)
                    


            self.plot_info['playout advantage'].append(sum(rewards))

            tmp = min(500, len(self.plot_info['playout advantage']))
            avg_return = sum(self.plot_info['playout advantage'][-tmp:-1])/tmp
            pbar.set_description(f"avg. return == {avg_return}")
        self.display_plots()
