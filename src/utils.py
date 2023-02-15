from typing import Tuple
import matplotlib.pyplot as plt
import torch

import numpy as np

def prepare_batch(states, actions, values, device='cpu'):
    '''Convert lists representing batch of transitions into pytorch tensors
    '''
    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
    values = torch.tensor(np.array(values), dtype=torch.float32, device=device)
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

