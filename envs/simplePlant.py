# -*- coding: utf-8 -*-
import gym
import numpy as np
from utils.Tester import Tester

# This class implements the notation of the gym library.
# we refer to https://www.gymlibrary.dev/ for further details.

class SimplePlant(gym.Env):

    def __init__(self, instance, completeDemand, seasonality = 1):
        
        super(SimplePlant, self).__init__()

        # Initial State:
        self.instance = instance
        self.tester = Tester()
        # Demand
        self.scenario_demand = completeDemand
        #horizon
        self.horizon = completeDemand.shape[1]
        #seasonality
        self.seas = seasonality
        #inventory
        self.initInventory = self.instance.inventory.copy()

        # State variable:
        self.reset()

    def _next_observation(self):
        self.demand = self.scenario_demand[:, self.current_step]

    def _take_action(self, action, inventory_level, demand):
        #simulation of the production step when the decision has been made.
        profit, item_produced = self.tester.produce(self.instance, action, demand)

        self.production = item_produced
        
        component_used = (item_produced.T).dot(self.instance.gozinto)
        # NB: element wise modification keeps reference
        for i in range(self.instance.n_components):
            inventory_level[i] = inventory_level[i] + action[i] - component_used[i]

        self.missed_demand = np.maximum(demand - item_produced, 0)
        #profit adjustment
        holdingCosts = np.sum(self.instance.holding_costs*inventory_level)
        lost_sales = np.sum(self.missed_demand*self.instance.lost_sales)
        profit -= (lost_sales + holdingCosts)


        return {
            'profit': profit,
            'holding_costs': holdingCosts,
            'lost_sales': lost_sales
        }


    def reset(self): 
        """
        Reset all environment variables important for the simulation.
            - Inventory
            - Demand_function
            - Current_step
        """
            
        self.current_step = 0
        self.instance.inventory = self.initInventory.copy()
        self.total_cost = {
            "profit": 0.0,
            "holding_costs": 0.0,
            "lost_sales": 0.0
        }
        self.production = [0.0] * self.instance.n_items
        self.action = np.zeros(self.instance.n_components)
        self.missed_demand = np.zeros(
            shape=(self.instance.n_items, )
        )

        self._next_observation()
        
        return {'demand':self.demand,'inventory': self.instance.inventory,'seasonalFactor': 0}

    def step(self, action):
        self.total_cost = self._take_action(
            action, self.instance.inventory, self.demand
        )
        self.action = np.array(action)
        reward = self.total_cost['profit']

        #Step
        self.current_step += 1
        done = self.current_step == self.horizon
        if not done:
            self._next_observation()
    
        info = {
            "profit": self.total_cost['profit'],
            "holding_costs": self.total_cost['holding_costs'],
            "lost_sales": self.total_cost['lost_sales'],
            "total_demand": np.sum(self.scenario_demand[:, self.current_step - 1]),
            "missed_demand": np.sum(self.missed_demand),
            "lost_sales": self.total_cost['lost_sales'],
            "holding_costs": self.total_cost['holding_costs'], 
            "production_costs": self.action.dot(self.instance.costs),
            "total_inventory": sum(self.instance.inventory)
        }

        #observation 
        obs = {'demand':self.demand,'inventory': self.instance.inventory, 'seasonalFactor': self.current_step % self.seas }

        # self.render() #if uncommented is useful to debug session. it plots for every steps all the passages.

        return obs, reward, done, info


    def render(self):
        # Render the environment to the screen
        print(f'Time: {self.current_step}')
        print(f'\t demand: {self.scenario_demand[:, self.current_step-1]}')
        print(f'\t item sold: { [f"{ele:.0f}" for ele in self.production] }')
        print(f'\t missed_demand: { self.missed_demand }')

        print(f'\t components produced: { [f"{ele:.0f}" for ele in self.action] }')
        print(f'\t inventory: {self.instance.inventory}')

        print(f"\t cost: ")
        print(f"\t \t profit: {self.total_cost['profit']:.2f}")
        print(f"\t \t holding_costs: {self.total_cost['holding_costs']:.2f}")
        print(f"\t \t production: { self.action.dot(self.instance.costs):.2f}")
