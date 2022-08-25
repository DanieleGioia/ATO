# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as grb
from solver.solverGurobi.atoG import AtoG
from scenarioTree import ScenarioTree
from scenarioReducer import *

class DummyScenarioReducer(Scenario_reducer):
    """
    Fake scenario reducer that is employed when the number of requested branch is just 1
    When this happen, the reduction boils down to the average of the available data
    """   
    def __init__(self, data):
        self.data = data

    def reduce(self, n):
        tmp = np.average(
            self.data,
            axis=1
        )
        demand_reduced = np.reshape(tmp,(tmp.size, 1))
        probs_reduced = [1]
        return demand_reduced, probs_reduced 


class AtoRPMultiStage(AtoG):
    """
    Standard version of the ATO problem, see the companion paper/thesis for the explicit model
    """
    def __init__(self, **setting):
        super().__init__(**setting)
        self.name = "RPMultiStage"
        self.branching_factors = setting['branching_factors']
        self.seas = 1 # no seasonality
        self.current = 0 # only needed if seasonal

    def seasonalize(self,seasonality):
        #saves the seasonality factors
        self.seas = seasonality
    
    def updateClock(self):
        # the clock is necessary to use the right data in case of multiple stages with seasonality
        self.current += 1

    def populate(self, instance, scenarios):
        items = range(instance.n_items)
        machines = range(instance.n_machines)

        FFreducers = []
        for i in range(len(self.branching_factors)): #the length of the horizon is given by the branching factors specification
            #selection of the data w.r.t. the seasonality. (We assume that we cannot use data from months with peaks of demand to decide on months with low demand)
            selected_data = scenarios[:,np.arange( (self.current + i) % self.seas ,len( scenarios[0,:]), self.seas)]
            #Reduction of the number of nodes according to a W2 fast forward scenario reducer 
            if self.branching_factors[i] > 1:
                FFreducers.append(
                    Fast_forward_W2(
                        selected_data
                    )
                )
            else: #here the reduction boils down to the average
                FFreducers.append(
                    DummyScenarioReducer(
                        selected_data
                    )
                )
        #scenario tree building
        scenario_tree = ScenarioTree(
            name='tree1',
            branching_factors=self.branching_factors,
            dim_observations=instance.n_items,
            initial_value=np.array([1]*instance.n_items),
            stoch_model=FFreducers
        )

        # ScenarioTree()
        nodes = range(scenario_tree.n_nodes)

        #initial inventory
        I_0 = np.array(instance.inventory)
        # model initialisation
        model = grb.Model(self.name)
        #production decision: number of components (root node)
        X = model.addMVar(
            shape=(instance.n_components),
            vtype=grb.GRB.CONTINUOUS,
            name='X'
        )
        #dummy decision made on future stages (one for each node of the tree)
        X_ms = model.addMVar(
            shape=(instance.n_components, scenario_tree.n_nodes),
            vtype=grb.GRB.CONTINUOUS,
            name='X_ms'
        ) 
        # sold items per node
        Y = model.addMVar(
            shape=(instance.n_items, scenario_tree.n_nodes),
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )
        # lost sales per node
        L = model.addMVar(
            shape=(instance.n_items, scenario_tree.n_nodes),
            vtype=grb.GRB.CONTINUOUS,
            name='L'
        )
        #inventory variable
        I = model.addMVar(
            shape=(instance.n_components, scenario_tree.n_nodes),
            vtype=grb.GRB.CONTINUOUS,
            name='I'
        )
        #profits 
        expr = sum(
            scenario_tree.nodes[n]['prob'] * instance.profits @ Y[:, n]
            for n in nodes
        )
        # lost sales cost
        expr -= sum(
            scenario_tree.nodes[n]['prob'] * instance.lost_sales @ L[:, n]
            for n in nodes
        )
        # components cost
        expr -= sum(
            scenario_tree.nodes[n]['prob'] * instance.costs[:] @ X_ms[:, n]
            for n in nodes
        )
        # holding costs
        expr -= sum(
            scenario_tree.nodes[n]['prob'] * instance.holding_costs[:] @ I[:, n]
            for n in nodes
        )
        model.setObjective(expr, grb.GRB.MAXIMIZE)

        # Capacity constraint for each machine
        model.addConstrs(instance.processing_time[:, m] @ X_ms[:, n] <= instance.availability[m] for m in machines for n in nodes)
        
        # Y bounds
        model.addConstrs(Y[j, n] + L[j, n] == scenario_tree.nodes[n]['obs'][j] for j in items for n in nodes)

        # Initial sondition
        model.addConstr(I[:, 0] == I_0[:], name="initial_condition")

        #root node definition
        model.addConstr(X == X_ms[:, 0], name="X_def")
        
        # Evolution
        for n in range(1, scenario_tree.n_nodes):
            parent = list(scenario_tree.predecessors(n))[0]
            model.addConstr(I[:, n] + instance.gozinto.T @ Y[:, n] == X_ms[:, parent] + I[:, parent], name="evolution")
        
        model.update()
        return model
