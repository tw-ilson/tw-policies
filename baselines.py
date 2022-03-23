from typing import Set, Dict

class AbstractBaseline():
    '''Abstract class defining a baseline for use with policy gradient methods'''

    def __init__(self) -> None:
        raise NotImplemented

    def __call__(self, state) -> float:
        raise NotImplemented

    def update(self, tau:Dict):
        raise NotImplemented

class TabularBaseline(AbstractBaseline):
    '''Calculates a baseline based on the expected return from a certain state'''

    def __init__(self) -> None:
        self.baselines = {}

    def __call__(self, state) -> float:
        self.baselines.setdefault(state, 0)

        return self.baselines[state]

    def update(self, trajectories: Set[Dict]):

        for tau in trajectories:
            sum( \
                abs(self.baselines[s_t] 




        


