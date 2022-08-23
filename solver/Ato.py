import abc
from abc import abstractmethod
import numpy as np

class Ato(object):
    """Base class for assemble-to-order problem"""

    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __init__(self, **setting):
        """
        Setting contains a dictionary with the parameters
        """
        self.name = None
        self.setting = setting

    def set_initial_inventory(self, initial_inventory):
        self.initial_inventory = initial_inventory

    @abstractmethod
    def solve(self, instance, scenarios: np.array):
        pass
