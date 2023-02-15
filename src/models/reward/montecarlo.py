from collections import OrderedDict
from typing import Callable
from models.reward import AbstractReturns

class MonteCarloReturns(AbstractReturns):
    """ This advantage function requires full monte-carlo style playouts of a policy in order to directly calculate the returns from each state in the given playout.
    """

    def __init__(self, gamma, normalize:bool=True) -> None:
        self.gamma = gamma
        self.returns_dict = {}
        self.normalize = normalize

    def update_baseline(self, tau):
        '''based on a playout, calculates the average expected return from all states encountered (inefficient)
        '''
        states, actions, rewards = tau
        self.returns_dict = self.reward_to_go(states, rewards)
        # assert not np.isnan(list(self.returns_dict.values())).any()

    def reward_to_go(self, states, rewards):
        """ Given a single full playout, implements top-down dynamic programming to compute the return following from each state encountered in the trajectory
        """
        trajectory = OrderedDict()

        t = len(rewards) - 1
        s_t = states[-1]
        trajectory[bytes(s_t)] = rewards[-1]
        while t > 0:
            t-=1
            s_nxt = s_t.copy()
            s_t = states[t]
            r = rewards[t]
            trajectory[bytes(s_t)] = r + self.gamma * trajectory[bytes(s_nxt)]

        if self.normalize:
            avg = sum(trajectory.values())/len(trajectory)
            for s in trajectory:
                trajectory[s] -= avg

        return trajectory

    def __call__(self, s):
        return self.returns_dict[bytes(s)] 

class MonteCarloBaselineReturns(MonteCarloReturns):
    """ This Advantage function requires full monte-carlo style playouts of a policy in order to directly calculate the returns from each state in the given playout.
    Here we introduce the notion of a baseline, which can be any function, that we subtract our returns 
    """

    def __init__(self, gamma, baseline:Callable) -> None:
        super().__init__(gamma)
        self.baseline = baseline

    def update_baseline(self, *playouts):
        super().update_baseline(*playouts)

        self.returns_dict = {s: self.returns_dict[s] - self.baseline(s) for s in self.returns_dict}

