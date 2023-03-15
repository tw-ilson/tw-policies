from collections import deque
from models.returns import AbstractReturns

class MonteCarloReturns(AbstractReturns):
    """Calculates an exact return value for each state based on the history of transitions
        Important: Needs a full playout
    """
    def __init__(self, gamma:float, normalize:bool):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def __call__(self, rewards):
        return self.reward_to_go(rewards)

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
                returns[s] -= avg

        return returns
