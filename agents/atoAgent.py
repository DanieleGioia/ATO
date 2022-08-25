from solver.Ato import Ato
from abc import abstractmethod

class AtoAgent():
    @abstractmethod
    def __init__(self, env, solver:Ato , observedDemand):
        # The solver identifies a policy of production. It will solve a particular ATO problem
        # on a particular env for a given observation of the demand
        pass

    @abstractmethod
    def get_action(self, obs):
        # The get action function generate the production decision w.r.t. the observed state of the system
        pass
