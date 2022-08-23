from solver.Ato import Ato
from abc import abstractmethod

class AtoAgent():
    @abstractmethod
    def __init__(self, env, solver:Ato , observedDemand):
        pass

    @abstractmethod
    def get_action(self, obs):
        pass
