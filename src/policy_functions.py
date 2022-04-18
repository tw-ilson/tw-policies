from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from cnn import CNN


class AbstractPolicyApproximator(ABC):
    '''Abstract class to represent a stochastic policy function approximator.
    '''

    def __init__(self, state_dim, action_space, alpha) -> None:
        self.state_dim = state_dim.shape
        self.action_space = action_space.n
        self.alpha = alpha
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def compute_score(self, tau):
        ''' Given a full playout of this policy, computes the score sum for the loss output of the policy approximation
        '''
        pass

    @abstractmethod
    def compute_step_score(self, s_t, a_t, r, log_prob, G):
        ''' Given a full playout of this policy, computes the score sum for the loss output of the policy approximation
        '''
        pass

    @abstractmethod
    def update_params(self, score):
        '''  Performs backwards pass to updates network parameters according to score calculation.
        Params
        ------
            score: the 'score function' as described in PG literature, typically computed using log probability of action scaled by 
    '''
        pass

    @abstractmethod
    def __call__(self, state:np.ndarray) -> np.ndarray:
        '''Given a state, return an action and an approximation of the log probability vector of the action space distribution
        '''
        pass

class LinearDiscretePolicyApproximator(AbstractPolicyApproximator):
    '''Approximates a policy using a Linear combination of the features of the state'''

    def __init__(self, state_dim, action_space, alpha) -> None:
        super().__init__(state_dim, action_space, alpha)
        self.state_dim = self.state_dim[0]

        #initialize weights matrix, single fully connected layer, no bias 
        self.weights = np.random.uniform(0, 1, size=(self.action_space, self.state_dim)).astype(np.float64)

        self.d_weights = np.zeros((self.state_dim, self.action_space))
        
    
    def forward(self, state):
        #Dense linear combination of state features
        hs = np.dot(self.weights, state)

        #SoftMax following output layer; numerically stable
        def log_softmax(x):
            assert not all(i == 0 for i in x)
            max_x = np.max(x)
            log_probs = x - max_x - np.log(np.sum(np.exp(x - max_x)))
            return log_probs

        #take the softmax function over hidden states
        log_probs = log_softmax(hs)

        return log_probs

    def __call__(self, state: np.ndarray) -> Tuple[int, np.ndarray]:

        log_probs = self.forward(state)

        #convert back from log space to discrete categorical using Gumbel-max trick:
        g = np.random.gumbel(0, .99, size=self.action_space)
        action = np.argmax(log_probs + g) #sample
        # probs = np.exp(log_probs)
        # action = np.random.choice(range(self.action_space), p=probs)

        return action, log_probs[action] - 1e-9 #add small value so no zero divide

    def compute_score(self, tau, G_tau):
        '''computes score with respect to whole playout
        '''

        score = 0
        for s_t, a_t, r, log_prob in tau:
            score += self.compute_step_score(s_t, a_t, r, log_prob, G_tau(s_t))
        return score

    def compute_step_score(self, s_t, a_t, r, log_prob, G):
            ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
                which follows a single linear combination layer of the state input.
            '''
            sm_probs = self.forward(s_t)

            #one-hot encode action
            act_mask = np.zeros(self.action_space)
            act_mask[a_t] = 1

            #the score computation
            score = -log_prob * G

            #The components of the gradient via chain rule
            SM =  sm_probs.reshape((-1,1))
            Jsm_dh = np.diagflat(sm_probs) - np.dot(SM, SM.T)
            dh_dw = s_t

            #gradient descent step 
            dL_dh = G * np.dot(Jsm_dh, act_mask) / score
            self.d_weights += np.outer(dh_dw, dL_dh)

            assert not np.isnan(self.d_weights).any()
            return score


    def update_params(self, score):
        #optimize using computed gradients
        
        self.weights = self.weights + self.alpha * self.d_weights.T
        self.d_weights = np.zeros((self.state_dim, self.action_space))

    def get_params(self):
        return self.weights
    

class NNDiscretePolicyApproximator(AbstractPolicyApproximator, nn.Module):
    '''Neural Network approximator for a categorical policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            hidden_size=64, 
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicyApproximator.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        if from_image:
            self.conv = CNN(self.state_dim)
            self.layers = nn.Sequential(
                    self.conv,
                    nn.Linear(self.conv.output_size, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_space),
                    )
        else:
            self.state_dim = self.state_dim[0]
            self.layers = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_space),
                    )

        self.optim = Adam(self.get_params(), lr=self.alpha )
        self.to(self.device)

    def __call__(self, state: np.ndarray) -> Tuple[int, torch.tensor]:
        probs = self.forward(torch.tensor(state).unsqueeze(0))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # print(type(log_prob))
        return action.item(), log_prob
        
    def compute_score(self, tau, G_tau):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        scores = []
        for s_t, a_t, r, log_prob in tau:
            score = -log_prob * G_tau(s_t)
            scores.append(score.unsqueeze(0))
        return torch.cat(scores).sum()

    def compute_step_score(self, s_t, a_t, r, log_prob, G):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        return -log_prob * G
    
    def update_params(self, score):
        '''updates network parameters according to score calculation.
        '''
        score.to(self.device)
        self.optim.zero_grad()
        score.backward(retain_graph=True)
        self.optim.step()

        #inplace detach so we can plot this
        score.detach_().cpu()

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32, device='cpu')
        x = self.layers(state)
        y = F.softmax(x, dim=-1).cpu()
        return y

    def get_params(self):
        return self.parameters()

class NNGaussianPolicyApproximator(AbstractPolicyApproximator, nn.Module):
    '''Neural Network approximator for a continuous policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            hidden_size=32, 
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicyApproximator.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)

        if not from_image:
            self.layers = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, 2)
                )
        else:
            # self.layers.append(CNN())
            raise NotImplemented

        self.optim = Adam(self.get_params(), lr=self.alpha )

    def __call__(self, state: np.ndarray) -> Tuple[int, torch.tensor]:
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
        
    def compute_score(self, s_t, a_t, log_prob, advantage):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        score = -log_prob * advantage
        # print(score)
        return score.unsqueeze(0)
    
    def update_params(self, scores):
        '''updates network parameters according to score calculation.
        '''
        score = torch.cat(scores).sum()
        self.optim.zero_grad()
        score.backward(retain_graph=True)
        self.optim.step()
        

    def forward(self, state):
        x = self.layers(torch.FloatTensor(state))
        return 

    def get_params(self):
        return self.parameters()
