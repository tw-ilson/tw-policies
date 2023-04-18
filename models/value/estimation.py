import torch
from torch import nn
import numpy as np
import  gym
from gym.spaces.utils import flatdim

from .montecarlo import MonteCarloReturns

from ...networks import FeedForward, CNN
from ...utils import action_mask, prepare_batch
from . import AbstractReturns

def TD_resid(v, vp, r, discount):
    v_target = discount * vp + r
    return v_target - v

def TDErr(v, vp, r, discount):
    ''' time-difference error between numeric value at t, and value at the following timestep
    '''
    err = TD_resid(v, vp, r, discount)
    return (err).mean()

#
# class ValueNetworkReturns(AbstractReturns, torch.nn.Module):
#     def __init__(self, state_space, n_hidden=1, hidden_size=64, lr=1e-3, gamma=0.98) -> None:
#         """Produces value estimation for state-action pairs 
#
#         Args:
#             input_shape (tuple, optional): Shape of the image (C, H, W). Should be square.
#             NOTES: have 2 predictions of direct joint values: (base and gripper), two predictions
#             of combinations of motors: (forward z values, and vertical position adjust)
#         """
#         super(ValueNetworkReturns, self).__init__()
#
#         # assert state_space.is_np_flattenable, "needs to be flattened for Neural Network"
#         self.gamma = gamma
#
#         self.d_state = flatdim(state_space)
#
#         self.feedforward = FeedForward(
#                 d_input=self.d_state,
#                 d_output=1,
#                 d_hidden=hidden_size,
#                 n_hidden=n_hidden)
#
#         self.mse = torch.nn.MSELoss()
#         self.optim = torch.optim.Adam(self.parameters(), lr=lr)
#
#     def __call__(self, state):
#         return self.forward(state)
#
#     def forward(self, state) -> torch.Tensor:
#         state = torch.as_tensor(state)
#         return self.feedforward(state)
#
#     @torch.no_grad()
#     def predict(self, state: torch.Tensor) -> torch.Tensor:
#         "Predicts 4d action for an input state, without storing gradient"
#         V = self.forward(state)
#         return V
#     
#     def optimize(self, s, td_err):
#         '''optimizes the value estimator
#         '''
#         self.optim.zero_grad()
#         loss = td_err 
#         loss.backward()
#         self.optim.step()

class QValueNetworkReturns(AbstractReturns, nn.Module):
    """Q-value network which computes a predicted return based on current state and action taken
    """

    def __init__(self, state_space, action_space, n_hidden=1, hidden_size=64, lr=0.001, gamma=0.98) -> None:
        super(QValueNetworkReturns, self).__init__()
        # assert state_space.is_np_flattenable, "state space needs to be flat for neural network"
        # assert action_space.is_np_flattenable, "action space needs to be flat for neural network"

        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.discrete_action = isinstance(action_space, gym.spaces.Discrete)

        self.d_state = flatdim(state_space)
        self.d_action = flatdim(action_space)

        self.affine1 = nn.Linear(self.d_state, self.d_state)
        self.feedforward = FeedForward(
                d_input= self.d_state + self.d_action,
                d_output=1,
                d_hidden=hidden_size,
                n_hidden=n_hidden,
                dropout=0.6,
                batchnorm=True)

        self.optim = torch.optim.Adam(self.feedforward.parameters(), lr=lr)
        self.to(device=self.device)

    def __call__(self, state, action):
        return self.forward(state, action)

    def forward(self, state, action) -> torch.Tensor:
        if self.discrete_action:
            action = action_mask(action, self.d_action)
        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        x = torch.cat((self.affine1(state), action), dim=1)
        return self.feedforward(x)

    @torch.no_grad()
    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        Q = self.forward(state, action)
        return Q

    def optimize(self, td_err):
        '''optimizes the q-value estimator
        '''
        self.optim.zero_grad()
        loss = td_err #* self.forward(s, a)
        loss.mean().backward()
        self.optim.step()
        del loss

    def td_err(self, s, a, r, sp, ap) -> float:
        '''Alias function; Calculates the SARSA td-err for the value network based on a transition
        '''
        return TDErr(self.predict(s, a), self.predict(sp, ap), r, self.gamma)

class AdvantageNetworkReturns(AbstractReturns, nn.Module):
    """ In the advantage estimation, we are calculating the difference between the reward + Q-value (state-action value) and the Value (state value). This represents the amount of 'advantage' in taking a *specific* action at a state. AKA the estimated TD-residual
    """ 
    def __init__(self, state_space, n_hidden=1, hidden_size=64, lr=1e-5, gamma=0.98) -> None:
        super(AdvantageNetworkReturns, self).__init__()
        self.d_state = flatdim(state_space)
        self.gamma = gamma
        self.V = FeedForward(
                    d_input=self.d_state, 
                    d_output=1, 
                    n_hidden=n_hidden, 
                    d_hidden=hidden_size)
        # self.montecarlo = MonteCarloReturns(gamma, normalize=False)
        self.entropy = 0
        self.optim = torch.optim.Adam(self.V.parameters(), lr)

    def __call__(self, s, r):
        return self.forward(s, r)

    def forward(self, states, rewards):
        states, rewards = prepare_batch(states, rewards)
        next_states = states[1:, :]
        states = states[:-1,:]
        v = self.V.forward(states).squeeze()
        vp = self.V.forward(next_states).squeeze()
        # out = TD_resid(v, vp, rewards, self.gamma)
        out = self.gamma * vp + rewards - v
        return out

    def optimize(self, advantage, entropy):
        """ The network output is already an error measure, so loss is simple mse loss
        """
        self.optim.zero_grad()
        self.entropy += entropy
        loss = 0.5 * advantage.pow(2) + 0.001 * self.entropy
        loss.mean().backward()
        self.optim.step()
        del advantage
        del loss
        # return loss

class GAEReturns(AbstractReturns, nn.Module):
    """ Generalized Advantage Estimator
    """

    def __init__(self, lam, state_space, n_hidden=1, hidden_size=64, lr=1e-5, gamma=0.98) -> None:
        """ Lambda is the hyperparameter introduced by GAE, to facilitate the tradeoff between the actual, high-variance rewards, and the estimated value function.
        """
        super(GAEReturns, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lam = lam
        self.d_state = flatdim(state_space)
        self.V = FeedForward(self.d_state, 1, n_hidden, hidden_size)
        self.montecarlo = MonteCarloReturns(gamma)
        self.gamma = gamma
        self.entropy = 0
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.V.parameters(), lr)
        self.to(device=self.device)

    def residuals(self, states, rewards):
        next_states = states[1:, :]
        states = states[:-1,:]
        v = self.V(states).squeeze()
        vp = self.V(next_states).squeeze()
        resid = self.gamma * vp + rewards - v
        return resid

    def __call__(self, states, rewards):
        states, rewards = prepare_batch(states, rewards)
        return self.forward(states, rewards)

    def forward(self, states, rewards):
        deltas = self.residuals(states, rewards)
        advantage = [] 
        for t in range(len(rewards)):
            tot = 0
            for l, delt in enumerate(deltas[t:]):
                tot += (self.gamma * self.lam)**l * delt 
                assert torch.is_tensor(tot)
            advantage.append(tot)
        advantage = torch.stack(advantage)
        assert(rewards.size() == advantage.size())
        return advantage

    # def optimize(self, loss):
    #     self.optim.zero_grad()
    #     loss.backward()
    #     self.optim.step()
    
    def optimize(self, states, rewards):
        """ Monte-Carlo error
        """
        self.optim.zero_grad()
        returns = self.montecarlo.reward_to_go(rewards)
        states, returns = prepare_batch(states, returns)
        v = self.V(states).squeeze()
        loss = self.loss(v, returns)
        loss.backward()
        self.optim.step()

    # def optimize(self, advantage, entropy):
    #     self.optim.zero_grad()
    #     self.entropy += entropy
    #     loss = 0.5 * advantage.pow(2) + 0.001 * self.entropy
    #     loss.mean().backward()
    #     self.optim.step()
    #     del advantage
    #     del loss
