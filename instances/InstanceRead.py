# -*- coding: utf-8 -*-
import numpy as np
from instances.Instance import Instance
from sampler.Sampler import Sampler
import json

class InstanceRead(Instance):
    """
    This class read a formatted instance scenario as dictionary.
    It adapts the tightness of the availability production w.r.t. the sampler
    """
    def __init__(self, setting: dict, sampler: Sampler , path_name: str ):
        self.name = 'Instance read class'
        self.sampler = sampler
        self.sim_setting = setting
        self.tightness = setting['tightness']
        self.set_seed(setting["seed"])
        #loading
        self._load(path_name)
        #machine availability
        self._getMachineAvailability()



    def _load(self, path_name):

        fp = open(path_name, 'r')
        data = json.load(fp)
        fp.close()
        self.n_items = int(data["n_items"])
        self.n_components = int(data["n_components"])
        self.n_machines = int(data["n_machines"])

        self.costs = np.array(data["costs"])
        self.profits = np.array(data["profits"])
        self.processing_time = np.array(data["processing_time"])
        self.gozinto = np.array(data["gozinto"])
        self.holding_costs = np.array(data["holding_costs"])
        self.lost_sales = np.array(data["lost_sales"]) 

    def _getMachineAvailability(self):
        # processing times of each component on each machine type (look at the size)
        mean_demand = self.sampler.get_sample_average_demand()
        component_requested = (self.gozinto.T).dot(mean_demand)
        self.availability = self.tightness * (self.processing_time.T).dot(component_requested)
        return self.availability

    def _getGozinto(self):
        return self.gozinto
    def _getProfit(self):
        return self.profits
    def _getCosts(self):
        return self.costs
    def _getHoldingCosts(self):
        return self.holding_costs
    def _getLostSales(self):
        return self.lost_sales
    def _getProcessingTime(self):
        return self.processing_time