# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as grb
from solver.solverGurobi.atoG import AtoG

class AtoRP(AtoG):
    """
    Standard Two-Stage stochastic LP model with recourse of the ATO problem,
    treated with the well-konwn Sampling Average Approximation (SAA).
    """
    def __init__(self,**setting):
        super().__init__(**setting)
        self.name = "RP"
        
    def populate(self, instance, scenarios):
        items = range(instance.n_items)
        machines = range(instance.n_machines)
        n_scenarios = scenarios.shape[1]
        I_0 = np.array(instance.inventory)
        # model initialisation
        model = grb.Model(self.name)
        # X are first stage solutions, common to every stochastic type of this problem
        X = model.addMVar(
            shape=instance.n_components,
            vtype=grb.GRB.CONTINUOUS,
            name='X'
        )
        #sold items in the second stage per scenario
        Y = model.addMVar(
            shape=(instance.n_items, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )
        # SAA plain probabilities 
        pi_s = 1.0 / (n_scenarios + 0.0)
        #
        expr = pi_s * sum(
            instance.profits @ Y[:, s]
            for s in range(n_scenarios)
        )#second stage profits 

        # first stage costs
        expr -= instance.costs[:] @ X

        model.setObjective(expr, grb.GRB.MAXIMIZE)

    
        model.addConstrs((Y[j, :] <= scenarios[j, :] for j in items), name="demand_constr") #number of sold items cannot be more than the demand
        model.addConstrs(instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines)  #machine availability constraint
        model.addConstrs(instance.gozinto.T @ Y[:, s] <= X + I_0 for s in range(n_scenarios)) #components and end items connection
        model.update()
        return model
