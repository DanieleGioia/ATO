from abc import abstractmethod
import numpy as np

class Sampler():
    """Base class for assemble-to-order scenarios of demand"""

    @abstractmethod
    def __init__(self, dict_distr: dict):
        """
        Dict_distr contains a dictionary with the parameters
        """    

    @abstractmethod
    def sample(self, n_scenarios: int = 1):
        pass

    def setInstanceOptions(self,dict_distr: dict):
        self.nItems = dict_distr["n_items"]
        self.set_seed(dict_distr["seed"])
        self.low = dict_distr['low']
        self.high = dict_distr['high']

    def get_sample_average_demand(self, n_scenarios=5000):
        demand = self.sample(n_scenarios)
        return np.average(demand, axis=1)
    
    def set_seed(self,seed = None):
        self.seed = seed
        np.random.seed(seed)
        
    def setNItems(self,howMany):
        self.nItems = howMany
    
    ###Methods to override
    def rescaleAdditive(self,seasonValue):
        raise ValueError('Additive rescaling not available in this sampler')
    def rescaleMultiplicative(self,seasonValue):
        raise ValueError('Multiplicative rescaling not available in this sampler')
    def revertAdditive(self,seasonValue):
        raise ValueError('Additive revert not available in this sampler')
    def revertMultiplicative(self,seasonValue):
        raise ValueError('Multiplicative revert not available in this sampler')

class SamplerIndependent(Sampler):
    """
    The independet sampler father class genereate an inizialilzation that 
    doesn't look at the gozinto. This latter would be necessary on family dependencies
    """  
    
    def __init__(self, dict_distr: dict):
        self.setInstanceOptions(dict_distr)
        self.name = 'Independent Sampler'


    @abstractmethod
    def sample(self, n_scenarios: int = 1):
        pass

class SamplerDependencies(Sampler):

    """
    The dependent sampler father class genereate an inizialilzation that 
    needs the gozinto information. This latter would be necessary on family dependencies
    """  
    
    def __init__(self, dict_distr: dict, dict_gozinto: dict):
        
        self.name = 'Dependencies Sampler'
        self.outcastItems = dict_gozinto['n_outcast_items']
        self.itemsPerFamily = dict_gozinto['n_items_per_family']
        self.setInstanceOptions(dict_distr)

    @abstractmethod
    def sample(self, n_scenarios: int = 1):
        pass