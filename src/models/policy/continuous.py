import numpy as np
import torch
from torch import nn
import gymnasium as gym
from models.policy import AbstractPolicy

class LinearGaussianPolicy(AbstractPolicy):
    '''Approximates a continuous policy using a Linear combination of the features of the state. Predicts a mean and standard deviation for each factor of the action space.
    '''
    def __init__(self, state_space, action_space, alpha) -> None:
        super().__init__(state_space, action_space, alpha)
        assert not isinstance(self.env.action_space, gym.spaces.Discrete), 'Discrete action space cannot be continuous'

        self.mu_weight = np.random.uniform(-1, 1, size=(self.action_dim, self.state_dim))
        self.sd_weight = np.ones(self.action_dim)

        # self.d_mu_weight = self.mu_weight.T.copy() * 0
        # self.d_sd_weight = self.sd_weight.T.copy() * 0

        self.EPS = 1e-6
    
    def forward(self, state):
        #Dense linear combination of state features
        mu = np.dot(self.mu_weight, state)

        #log-sd is based on pure weights
        sd = np.exp(self.sd_weight) #- logsumexp(self.sd_weight))
        sd = np.clip(sd, self.EPS, 2)

        return mu, sd

    def pdf(self, state):
        '''The gaussian log-probability density calculation
        '''
        mu, sd = self.forward(state)
        log_probs = lambda x: -0.5 * (x - mu)/sd**2 - np.log(sd * (2 * np.pi)**(0.5))
        return log_probs

    def __call__(self, state: np.ndarray):
        #sample randomly from gaussian distribution
        mu, sd = self.forward(state)
        dev = np.random.multivariate_normal(mu, np.diag(sd))

        act = mu + dev
        
        assert len(act) == self.action_dim
        return act

    def score(self, s, a, v):
        ''' In this version of the implementation, computes the gradient wrt to the output of the softmax function, 
            which follows a single linear combination layer of the state input.
        '''
        def step_score(s_t, a_t, v_t):
            mu, sd = self.forward(s_t)

            mu_grad = v_t * np.outer(s_t, (a_t - mu))/(sd**2)
            sd_grad = ((a_t - mu)**2 - sd**2)/ sd**3
            return np.array([mu_grad, sd_grad])
        
        scores = np.array([step_score(s_t, a_t, v_t) for s_t, a_t, v_t in zip(s, a, v)])
        return np.mean(scores[:, 0], axis=0), np.mean(scores[:, 1], axis=0)

    def optimize(self, score):
        #optimize using computed gradients
        self.mu_weight = self.mu_weight + self.alpha * score[0].T
        self.sd_weight = self.sd_weight + self.alpha * score[1]

        self.mu_weight = np.nan_to_num(self.mu_weight)
        self.sd_weight = np.nan_to_num(self.sd_weight)

    def get_params(self):
        return [self.mu_weight, self.sd_weight]

class NNGaussianPolicy(AbstractPolicy, nn.Module):
    '''Neural Network approximator for a continuous policy distribution.
    '''

    def __init__(self,
            state_space,
            action_space,
            hidden_size=64, 
            alpha:float=1e-3,
            from_image:bool=False):

        AbstractPolicy.__init__(self, state_space, action_space, alpha)
        nn.Module.__init__(self)
        assert not isinstance(action_space, gym.spaces.Discrete), 'Discrete action space cannot be continuous'


        if from_image:
            self.state_dim = self.state_space.shape
            self.conv = CNN(self.state_dim)
            self.mu = nn.Sequential(
                    self.conv,
                    nn.Linear(self.conv.output_size, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_dim),
                    )
        else:
            self.mu = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(True),  
                    nn.Linear(hidden_size, self.action_dim)
                    )

        # self.sigma = nn.Linear(hidden_size, self.action_dim)
        self.sigma = torch.ones(self.action_dim, dtype=torch.float32, requires_grad=True)

        self.optim = Adam(list(self.get_params()) + [self.sigma], lr=self.alpha )

    def pdf(self, state):
        mu, sd = self.forward(torch.tensor(state))
        dist = torch.distributions.Normal(mu, sd)
        return dist

    def __call__(self, state: np.ndarray):
        dist = self.pdf(state)
        action = dist.rsample()
        return action.detach().numpy()
    
    def score(self, s, a, v):
        ''' Computes the score function of the policy gradient with respect to the parameters
        '''
        states, actions, values = prepare_batch(s, a, v)
        dist = self.pdf(states)
        return torch.mean(-dist.log_prob(actions) * values)
    
    def optimize(self, score):
        '''updates network parameters according to score calculation.
        '''
        score.backward(retain_graph=True)
        self.optim.step()
        self.optim.zero_grad()
        score.detach_()

    def forward(self, state):
        EPS = 1e-6
        MAX = 3
        return self.mu(torch.FloatTensor(state)), torch.clip(torch.exp(self.sigma), EPS, MAX)

    def get_params(self):
        return self.parameters()
