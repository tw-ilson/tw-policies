from typing import Tuple
import matplotlib.pyplot as plt
import torch
import numpy as np
import gym
from gym.spaces.utils import flatten
import tracemalloc

def action_mask(a_t, action_dim):
    '''Discrete actions spaces only. Works with batch
    '''
    a_t = np.asarray(a_t, dtype=int)
    nbatch = a_t.shape[0]
    mask = np.zeros((nbatch, action_dim))
    mask[np.arange(nbatch), a_t] = 1
    return mask

def prepare_batch(*args, device='cpu'):
    '''Convert lists representing batch of transitions into pytorch tensors
    '''
    return [torch.as_tensor(p, dtype=torch.float32, device=device) for p in args]

def logsumexp(x):
    '''Trick to compute expontial without float overflow
    '''
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def plot_curves(**curves):
    print(len(curves))
    f, axs = plt.subplots(len(curves), 1, figsize=(7,2.5))
    # axs = [axs] if not isinstance(axs, list) else axs
    plt.subplots_adjust(hspace=0.3)
    W = 12 # smoothing window

    [a.clear() for a in axs]
    for i, name in enumerate(curves):
        # print(name, curves[name].shape)
        if torch.is_tensor(curves[name][0]):
            curves[name] = torch.tensor(curves[name], device='cpu').numpy()
        if len(curves[name]) > 0:
            axs[i].plot(np.convolve(curves[name], np.ones(W)/W, 'valid'))
        axs[i].set_xlabel('steps')
        axs[i].set_ylabel(name)
    plt.show()

