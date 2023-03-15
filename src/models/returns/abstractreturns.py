from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from models.cnn import CNN

class AbstractReturns(ABC):
    """ A policy gradient is only useful in the context of the state from which it was produced, and more generally, the utility of the playout trajectory from which it came, in terms of real success or failure. This data structure creates an abstraction for the problem of advantage estimation, as we get into more sophisticated policy gradient methods. There is not much static commanaility between 
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        self.debug = {}
        super().__init__()
        
    @abstractmethod
    def __call__(self, **kwargs):
        """Get the expected returns for a single transition
        """
        raise NotImplemented

    # @abstractmethod
    # def update_baseline(self, tau):
    #     """Update the baseline for the return function based on tau playouts data received.
    #     """
    #     pass
