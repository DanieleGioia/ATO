# -*- coding: utf-8 -*-
from copy import error
import numpy as np
import matplotlib.pyplot as plt
from instances.Instance import Instance
from sampler.Sampler import Sampler

class InstanceRandom(Instance):
    """
    This class generate an instance scenario completely random. All the necessary 
    parameters are listed in the dictionary of the setting.
    """
    def __init__(self, setting: dict , sampler: Sampler):
        """
        Setting contains a dictionary with the parameters
        """
        self.name = 'Instance random class'
        self.sampler = sampler
        self.sim_setting = setting
        super().set_seed(setting["seed"])
        # SETTING GENERAL PARAMETERS
        self.n_items = setting['n_items']
        self.n_components = setting['n_components']
        self.n_machines = setting['n_machines']
        self.gozintoParams = setting['dict_gozinto']
        self.inventory = np.array(setting['initial_inventory'])
        self.profit_margin_low = setting['profit_margin_low']
        self.profit_margin_medium = setting['profit_margin_medium']
        self.profit_margin_high = setting['profit_margin_high']
        self.perc_low_margin_item = setting['perc_low_margin_item']
        self.perc_medium_margin_item = setting['perc_medium_margin_item']
        self.processing_time_interval = setting['processing_time_interval']

        self.tightness = setting['tightness']
        self.component_cost = setting['component_cost']
        self.holding_costsPerc = setting['holding_costs']
        self.lost_SalesPerc = setting['lost_sales']

        #settings domain control
        self.__checkDomain()

        #generation of the instance
        self._getProcessingTime()
        self._getCosts()
        self._getHoldingCosts()
        self._getGozinto()
        self._getProfit()
        self._getLostSales()
        self._getMachineAvailability()


    def _getProcessingTime(self):
        # GENERATE PROCESSING TIMES
        self.processing_time = np.random.randint(
            low=self.processing_time_interval[0],
            high=self.processing_time_interval[1],
            size=(self.n_components, self.n_machines)
        )

    def _getCosts(self):
        #Generate components cost
        self.costs = np.sum(self.processing_time, axis=1) * np.random.uniform(0.8, 1.2, size=(self.n_components))
        self.costs = np.around(
            self.component_cost[0] + (self.component_cost[1] - self.component_cost[0]) * self.costs / max(self.costs),
            decimals=2
        )

    def _getHoldingCosts(self):
        # GENERATE HOLDING COSTS
        # costant
        # self.holding_costs = self.holding_costsPerc * np.mean(self.costs)
        self.holding_costs = self.holding_costsPerc * self.costs

    def _getProfit(self):
        # GENERATE PROFITS
        self.profits = np.zeros((self.n_items, ))
        perc_threshold = self.perc_low_margin_item + self.perc_medium_margin_item
        for j in range(self.n_items):
            # here we calculate the costs of making the product j (for every product available)
            for i in range(self.n_components):
                self.profits[j] += self.gozinto[j, i] * self.costs[i]
            # then we look at the margin and draw randomly from the selected interval (low, medium or high)
            # low margin interval
            if j < self.n_items * self.perc_low_margin_item:
                rnd_nmb = np.random.uniform(
                    self.profit_margin_low[0],
                    self.profit_margin_low[1]
                )
            # medium margin interval
            elif j < self.n_items * perc_threshold:
                rnd_nmb = np.random.uniform(
                    self.profit_margin_medium[0],
                    self.profit_margin_medium[1]
                )
            # high margin
            else:
                rnd_nmb = np.random.uniform(
                    self.profit_margin_high[0],
                    self.profit_margin_high[1]
                )
            # we add the percentage margin to the costs to obtain the profit (it is a revenue to be precise)
            self.profits[j] *= (1.0 + rnd_nmb)
        self.profits = np.around(self.profits, decimals=2)

    def _getLostSales(self):
        # GENERATE LOST SALES
        self.lost_sales = self.lost_SalesPerc * self.profits


    def _getMachineAvailability(self):
        # processing times of each component on each machine type (look at the size)
        mean_demand = self.sampler.get_sample_average_demand()
        component_requested = (self.gozinto.T).dot(mean_demand)
        self.availability = self.tightness * (self.processing_time.T).dot(component_requested)
        return self.availability

    def plotGozinto(self):

        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 12}
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick',labelsize=15)
        plt.rc('font', **font)
        plt.rc('axes', linewidth=2)

        fig = plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
        data_masked = np.ma.masked_where(
            self.gozinto == 0, self.gozinto
        )
        cax = plt.imshow(data_masked, interpolation = 'none', vmin = 1)

        fig.colorbar(cax)
        plt.xlabel('Components',fontsize=18,weight='bold')
        plt.ylabel('Items',fontsize=18,weight='bold')
        plt.savefig("./results/gozmatrix.png")
        plt.close()

    def _getGozinto(self):
        self.gozinto = np.zeros((self.n_items, self.n_components))
        # particular style of gozinto
        
        if self.gozintoParams['name'] == "nested":
            self.gozinto = np.tril(
                np.ones((self.n_items, self.n_components))
            )
        elif self.gozintoParams['name'] == "W_style":
            for i in range(self.n_items):
                self.gozinto[i][i] = 1
                self.gozinto[i][min(i + 1, self.n_components-1)] = 1
        elif self.gozintoParams['name'] == "M_style":
            print("M_style not supported")
            quit()
        elif self.gozintoParams['name'] == "standard":
            # OUTCAST
            start_row = self.n_items - self.gozintoParams['n_outcast_items']
            end_row = self.n_items
            self.gozinto[ start_row:end_row , :] = np.random.binomial(
                1,
                self.gozintoParams['p_outcast_component'],
                size = (end_row - start_row,self.n_components)
            )
            start_row = 0
            start_col = 0
            # ITEMS IN FAMILY
            for i, num_fam in enumerate(self.gozintoParams['n_items_per_family']):
                # COMMON COMPONENTS
                end_row = start_row + num_fam
                end_col = start_col + self.gozintoParams['n_components_per_family'][i]
                self.gozinto[ start_row:end_row , start_col:start_col+self.gozintoParams['n_common_components_per_family']] = np.random.randint(
                    self.gozintoParams['factor'][0],
                    self.gozintoParams['factor'][1],
                    size=(end_row - start_row, self.gozintoParams['n_common_components_per_family'])
                ) 
                # SPECIFIC COMPONENT
                start_col_spec = start_col+self.gozintoParams['n_common_components_per_family']
                end_col_spec = start_col_spec + num_fam
                self.gozinto[ start_row:end_row , start_col_spec:end_col_spec] = np.diag(
                    np.random.randint(
                        self.gozintoParams['factor'][0],
                        self.gozintoParams['factor'][1],
                        num_fam
                    )
                )
                if end_col_spec < end_col:
                    # ramaining specific components are set randomly
                    for j in range(end_col_spec, end_col):
                        rnd_row = np.random.randint(start_row, end_row)
                        self.gozinto[rnd_row, j] = np.random.randint(
                            self.gozintoParams['factor'][0],
                            self.gozintoParams['factor'][1]
                        )
                start_row = end_row
                start_col = end_col



    def __checkDomain(self):
        if self.n_items <= 0 or (not isinstance(self.n_items, int)):
            raise ValueError('n_items must be positive and integer') 
        if self.n_components <= 0 or (not isinstance(self.n_components, int)):
            raise ValueError('n_components must be positive and integer') 
        if self.n_machines <= 0 or (not isinstance(self.n_machines, int)):
            raise ValueError('n_machines must be positive and integer')
        if not all(isinstance(self.profit_margin_low[i],float) for i in range(2)):
            raise ValueError('Error on the margin definition')
        if not all(isinstance(self.profit_margin_medium[i],float) for i in range(2)):
            raise ValueError('Error on the margin definition')
        if not all(isinstance(self.profit_margin_high[i],float) for i in range(2)):
            raise ValueError('Error on the margin definition')
        if not all(self.profit_margin_low[i] >=0 for i in range(2)):
            raise ValueError('Margins must be non-negative')
        if not all(self.profit_margin_medium[i] >= 0 for i in range(2)):
            raise ValueError('Margins must be non-negative')
        if not all(self.profit_margin_high[i] >= 0 for i in range(2)):
            raise ValueError('Margins must be non-negative')
        if (self.perc_low_margin_item < 0 or self.perc_low_margin_item>1):
            raise ValueError('Percentages must range from 0 to 1')
        if (self.perc_medium_margin_item < 0 or self.perc_medium_margin_item>1):
            raise ValueError('Percentages must range from 0 to 1')
        if not all(self.component_cost[i] > 0 for i in range(2)):
            raise ValueError('Costs must be positive')
        if self.tightness < 0:
            raise ValueError('tightness must be positve')