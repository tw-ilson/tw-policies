from collections import deque
from . import AbstractReturns

class MonteCarloReturns(AbstractReturns):
    """Calculates an exact return value for each state based on the history of transitions
        Important: Needs a full playout
    """
    def __init__(self, gamma:float, normalize:bool=False):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize
        # self.returns = 0

    def reset(self):
        self.returns = 0

    def __call__(self, state):
        return self.returns(state)

    def reward_to_go(self, rewards):
        '''based on a playout, calculates the average expected return from all states encountered
        '''
        R = 0
        returns = deque()
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)

        # re-center data to help normalize gradient
        if self.normalize:
            avg = sum(returns)/len(returns)
            for s in returns:
                s -= avg
        self.returns = returns
        return returns

class BaselineMCReturns(MonteCarloReturns):
    def __init__(self, baseline, gamma: float=0.98, normalize: bool = False):
        super().__init__(gamma, normalize)
        self.baseline = baseline
        
