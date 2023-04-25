from abc import ABC, abstractmethod

class AbstractReturns(ABC):
    """ A shallow interface for representing a Value function / Q-Value function / Return function / Advantage function. 
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
    @abstractmethod
    def __call__(self, **kwargs):
        """Get the expected returns for a single transition
        """
        raise NotImplemented
