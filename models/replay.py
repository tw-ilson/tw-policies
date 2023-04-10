from typing import Tuple, Optional
import numpy as np

class ReplayBuffer:
    def __init__(self,
                 size: int,
                 state_shape: Tuple[int],
                 action_shape: Tuple[int],
                 continuous=False
                ) -> None:
        '''Replay Buffer that stores transitions (s,a,r,sp,d) and can be sampled
        for random batches of transitions

        Parameters
        ----------
        size
            number of transitions that can be stored in buffer at a time (beyond
            this size, new transitions will overwrite old transitions)
        state_shape
            shape of state image (H,W,C), needed to initialize data array
        action_shape
            shape of action (2,) since action is <px, py>, dtype=int
        '''
        self.length = 0
        self.size = size
        self._next_idx = 0
        self.data = {'state' : np.zeros((size, *state_shape), dtype=np.float32),
                     'action' : np.zeros((size, *action_shape), dtype=np.float32 if continuous else np.uint8),
                     'reward' : np.zeros((size), dtype=np.float32),
                     'next_state' : np.zeros((size, *state_shape), dtype=np.float32),
                     #'next_action': np.zeros((size, *action_shape), dtype=np.float32)
                    }
    def add_transition(self, s: np.ndarray, a: np.ndarray, r: float,
                       sp: np.ndarray) -> None:
        '''Add single transition to replay buffer, overwriting old transitions
        if buffer is full
        '''
        self.data['state'][self._next_idx,:] = s
        self.data['action'][self._next_idx, :] = a
        self.data['reward'][self._next_idx] = r
        self.data['next_state'][self._next_idx, :] = sp

        self.length = min(self.length + 1, self.size)
        self._next_idx = (self._next_idx + 1) % self.size

    def sample(self, batch_size: int) -> Tuple:
        '''Sample a batch from replay buffer.

        Parameters
        ----------
        batch_size
            number of transitions to sample
        '''
        idxs = np.random.randint(self.length, size=batch_size)

        s, a, r, sp = [self.data[k][idxs] for k in self.data.keys()]

        return s, a, r, sp

    def __len__(self):
        return self.length

