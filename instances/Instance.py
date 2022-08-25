from abc import abstractmethod
from sampler.Sampler import Sampler
import matplotlib.pyplot as plt
import numpy as np
import json
import time

class Instance():
    """Base class for assemble-to-order intances"""


    @abstractmethod
    def __init__(self, setting: dict, sampler: Sampler):
        """
        Setting contains a dictionary with the parameters
        The sampler is required to set the machine availability
        """
        self.name = None
        self.sim_setting = setting
        self.set_seed(setting["seed"])

    @abstractmethod
    def _getGozinto(self):
        #generate gozinto matrix
        pass

    @abstractmethod
    def _getProfit(self):
        # generate profits of the items
        pass

    @abstractmethod
    def _getCosts(self):
        # generate costs of the components
        pass

    @abstractmethod
    def _getHoldingCosts(self):
        # generate holddingCosts of the components
        pass

    @abstractmethod
    def _getLostSales(self):
        # generate costs of the lost sales of the items
        pass

    @abstractmethod
    def _getProcessingTime(self):
        # generate processing time of the components
        pass

    @abstractmethod
    def _getMachineAvailability(self) :
        # processing times of each component on each machine type
        pass

    def set_seed(self,seed = None):
        #set the seed of the psuedo-random generators
        self.seed = seed
        np.random.seed(seed)

    def plotGozinto(self):
        plt.matshow(self.gozinto)
        plt.xlabel('components')
        plt.ylabel('items')
        plt.savefig("./gozmatrix.pdf")
        plt.close()

    def print_on_file(self, name_instance=None):
        payload = {
            "costs": self.costs.tolist(),
            "profits": self.profits.tolist(),
            "processing_time": self.processing_time.tolist(),
            "gozinto": self.gozinto.tolist(),
            "n_items": self.n_items,
            "n_components": self.n_components,
            "n_machines": self.n_machines,
            "holding_costs": self.holding_costs.tolist(),
            "lost_sales":self.lost_sales.tolist(),
        }
        if name_instance:
            file_name = './inst_{}.json'.format(
                name_instance
            )
        else:
            file_name = './inst_{}.json'.format(
                int(time.time())
            )
        with open(file_name, 'w') as outfile:
            json.dump(
                payload, outfile,
                indent=2,
                sort_keys=True
            )