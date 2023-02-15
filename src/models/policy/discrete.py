import numpy as np
import torch
from torch import nn
from models.policy import AbstractPolicy
from models.cnn import CNN
from utils import logsumexp, prepare_batch

class LinearDiscretePolicy(AbstractPolicy):
    '''Approximates a policy using a Linear combination of the features of the state'''

    def __init__(self, state_space, action_space, alpha) -> None:
        super().__init__(state_space, action_space, alpha)
        #initialize weights matrix, single fully connected layer, no bias 
        self.weights = np.random.uniform(0, 1, size=(self.action_dim, self.state_dim)).astype(np.float64)
        self.d_weights = self.weights.T.copy() * 0

    def pdf(self, state):
        ''' The Log-Softmax produces categorical distribution
        '''
        x = self.forward(state)
        assert not all(i == 0 for i in x)
        log_probs = x - logsumexp(x)
        return log_probs
    
    def forward(self, state):
        #Dense linear combination of state features
        return np.dot(self.weights, state)

    def __call__(self, state: np.ndarray):

        log_probs = self.pdf(state)

        #convert back from log space to discrete categorical using Gumbel-max trick:
        g = np.random.gumbel(0, .99, size=self.action_dim)
        action = np.argmax(log_probs + g) #sample

        return action

    def score(self, s, a, v):
            ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
                which follows a single linear combination layer of the state input.
            '''
            def step_score(s_t, a_t):
                assert len(s_t) == self.state_dim

                sm_probs = self.pdf(s_t)
                
                #one-hot encode action
                act_mask = np.zeros(self.action_dim)
                act_mask[int(a_t)] = 1

                phi = np.outer(s_t, np.ones(self.action_dim))
                return phi - np.nan_to_num(
                        np.outer(s_t, np.exp(sm_probs * act_mask)))

            score = np.sum([v_t * step_score(s_t, a_t) for s_t, a_t, v_t in zip(s, a, v)], axis=0)
            return score


    def optimize(self, score):
        #optimize using computed gradients
        
        self.weights = self.weights + self.alpha * score.T
        self.d_weights = np.zeros((self.state_dim, self.action_dim))

    def get_params(self):
        return self.weights

class NNDiscretePolicy(AbstractPolicy, nn.Module):
    '''Neural Network approximator for a categorical policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            hidden_size=64, 
            alpha:float=1e-3,
            ):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        AbstractPolicy.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)

        # if from_image:
        #     self.state_dim = self.state_space.shape
        #     self.conv = CNN(self.state_dim)
        #     self.layers = nn.Sequential(
        #             self.conv,
        #             nn.Linear(self.conv.output_size, hidden_size),
        #             nn.Dropout(0.6), # To help normalize, prevent redundancy
        #             nn.ReLU(True),  
        #             nn.Linear(hidden_size, self.action_dim),
        #             )
        # else:
        self.layers = nn.Sequential(
                nn.Linear(self.state_dim, hidden_size),
                nn.Dropout(0.6), # To help normalize, prevent redundancy
                nn.ReLU(True), 
                nn.Linear(hidden_size, self.action_dim),
                )

        #include some very small noise
        self.epsilon = np.finfo(np.float32).eps.item()

        self.optim = torch.optim.Adam(self.get_params(), lr=self.alpha)
        self.to(self.device)

    def forward(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        x = self.layers(s)
        y = nn.functional.softmax(x, dim=-1)
        return y

    def pdf(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        return dist

    def __call__(self, state: np.ndarray):
        dist = self.pdf(state)
        action = dist.sample()
        return action.item()
        
    def score(self, s, a, v):
        ''' Computes the score function of the policy gradient with respect to the loss output of this neural Network; 
        '''
        s, a, v = prepare_batch(s, a, v, device=self.device)
        #v = v.clone().detach()
        dist = self.pdf(s)
        return torch.mean(-dist.log_prob(a) * v)
    
    def optimize(self, score):
        '''updates network parameters according to score calculation.
        '''
        score.backward(retain_graph=True)
        #scale gradient by advantage 
        # self.optim.step()
        # self.optim.zero_grad()
        score.detach_()

    def get_params(self):
        return self.parameters()
