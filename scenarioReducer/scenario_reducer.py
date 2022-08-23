from abc import abstractmethod

class Scenario_reducer():
    '''
    Abstract class for a scenario reducer that,
    given a fixed n, reduces an original set of scenarios
    with cardinality N to a smaller one of cardinality n.

    Different strategies can vary w.r.t.:
    -The metrics (Wass 1,2,inf..)
    -The selection order (Fast forward, backward, simultaneous backward)

    We assume a discrete uniform initial distribution on the scenarios.
    It can be generalized by appending a new value in the init of another sub-class

    For further details, please refer to:
    [1] Heitsch, Holger, and Werner Römisch. "Scenario reduction algorithms in stochastic programming." Computational optimization and applications 24.2-3 (2003): 187-206.
    '''
    @abstractmethod
    def __init__(self, initialSet):
        """
        The initial set is assumed to have a dimension d x N,
        where N is the number of the initial set of scenarios
        """    
    @abstractmethod
    def reduce(self, n_scenarios: int = 1):
        """
        reduces the initial set of scenarios
        """   
        pass
