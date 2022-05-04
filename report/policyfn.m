class AbstractPolicy(ABC):
    '''Abstract class to represent a stochastic policy function approximator.
    '''

    def __init__(self, state_space, action_space, alpha) -> None:
        self.state_space = state_space
        self.state_dim = gym.spaces.utils.flatdim(state_space)
        self.action_space = action_space
        self.action_dim = gym.spaces.utils.flatdim(action_space)
        self.alpha = alpha 

        self.plot_info = {}

    @abstractmethod
    def log_probability_density(self, *args):
    	'''Defines the PDF of the distributon. NOTE: not explicitly necessary for the PG
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def episode_score(self, tau):
        ''' Given a full playout of this policy, computes the score sum for the loss output of the policy approximation
        '''
        pass

    @abstractmethod
    def score(self, s_t, a_t):
        ''' Given a full playout of this policy, computes the score sum for the loss output of the policy approximation
        Params
        ------
            s_t: state at time t
            a_t: action taken from s_t
        '''
        pass

    @abstractmethod
    def optimize(self, score, advantage) -> None:
        '''  Performs backwards pass to updates network parameters according to score calculation.
        Params
        ------
            score: the 'score function' as described in PG literature, typically computed using log probability of action scaled by 
            advantage: the advantage of the state or sequence from which the score was computed.
        '''
        pass

    @abstractmethod
    def __call__(self, state:np.ndarray):
        '''Given a state, return an action and an approximation of the log probability vector of the action space distribution
        '''
        pass
