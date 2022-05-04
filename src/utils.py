from typing import Tuple
import matplotlib.pyplot as plt
import torch

import numpy as np

def prepare_batch(states, actions, values):
    '''Convert lists representing batch of transitions into pytorch tensors
    '''
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    return states, actions, values

def logsumexp(x):
    '''Trick to compute expontial without float overflow
    '''
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def plot_curves(**curves):
    print(len(curves))
    f, axs = plt.subplots(1, len(curves), figsize=(7,2.5))
    # axs = [axs] if not isinstance(axs, list) else axs
    plt.subplots_adjust(wspace=0.3)
    W = 12 # smoothing window

    [a.clear() for a in axs]
    for i, name in enumerate(curves):
        # print(name, curves[name].shape)
        if len(curves[name]) > 0:
            axs[i].plot(np.convolve(curves[name], np.ones(W)/W, 'valid'))
        axs[i].set_xlabel('steps')
        axs[i].set_ylabel(name)

class ReplayBuffer:
    def __init__(self,
                 size: int,
                 state_shape: Tuple[int],
                 action_shape: Tuple[int],
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
        self.data = {'state' : np.zeros((size, *state_shape), dtype=np.uint8),
                     'action' : np.zeros((size, *action_shape), dtype=np.int8),
                     'next_state' : np.zeros((size, *state_shape), dtype=np.uint8),
                     'reward' : np.zeros((size), dtype=np.float32),
                     'done' : np.zeros((size), dtype=np.bool8),
                    }
        self.length = 0
        self.size = size
        self._next_idx = 0

    def add_transition(self, s: np.ndarray, a: np.ndarray, r: float,
                       sp: np.ndarray, d: bool) -> None:
        '''Add single transition to replay buffer, overwriting old transitions
        if buffer is full
        '''
        self.data['state'][self._next_idx] = s
        self.data['action'][self._next_idx] = a
        self.data['reward'][self._next_idx] = r
        self.data['next_state'][self._next_idx] = sp
        self.data['done'][self._next_idx] = d

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

        keys = ('state', 'action', 'reward', 'next_state', 'done')
        s, a, r, sp, d = [self.data[k][idxs] for k in keys]

        return s, a, r, sp, d

    def __len__(self):
        return self.length

