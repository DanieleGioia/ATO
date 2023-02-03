# -*- coding: utf-8 -*-
from solver.solverGurobi.atoG_multi import AtoG_multi
import gurobipy as grb
import numpy as np


class AtoRP_multi(AtoG_multi):
    """
    ATO problem with recourse
    SAA methodology
    This multi version includes Lost Sales and Holding Costs
    """
    def __init__(self,**setting):
        super().__init__(**setting)
        self.name = "RP_multi"
        
    def populate(self, instance, scenarios, present_demand):
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
        I = model.addMVar(
            shape=instance.n_components,
            vtype=grb.GRB.CONTINUOUS,
            name='I'
        )
        #Z are the components not sold
        Z = model.addMVar(
            shape=(instance.n_components, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='Z'
        )
        #production variable
        Y = model.addMVar(
            shape=(instance.n_items, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )
        #production variable
        Y_0 = model.addMVar(
            shape=instance.n_items,
            vtype=grb.GRB.CONTINUOUS,
            name='Y_0'
        )
        #lost sale variable
        L = model.addMVar(
            shape=(instance.n_items, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='L'
        )
        #production variable
        L_0 = model.addMVar(
            shape=instance.n_items,
            vtype=grb.GRB.CONTINUOUS,
            name='L_0'
        )
        # crude Montecarlo is deployed (room for generalisation)
        pi_s = 1.0 / (n_scenarios + 0.0)
        #
        expr = pi_s * sum(
            instance.profits @ Y[:, s]
            for s in range(n_scenarios)
        )

        expr-= pi_s * sum(
            instance.lost_sales @ L[:, s]
            for s in range(n_scenarios)
        )


        expr-= pi_s *  sum( (instance.holding_costs*np.ones(instance.n_components) ) @ Z[:, s]
            for s in range(n_scenarios)
        )

        # first stage costs
        expr -= instance.costs[:] @ X
        expr -= instance.lost_sales[:] @ L_0
        expr += instance.profits[:] @ Y_0

        model.setObjective(expr, grb.GRB.MAXIMIZE)

        # Capacity constraint for each machine
        model.addConstrs((Y[j, :] + L[j, :] == scenarios[j, :] for j in items), name="demand_constr")
        model.addConstr((Y_0 + L_0 == present_demand), name="demand_constr")
        model.addConstrs(instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines)
        model.addConstrs(instance.gozinto.T @ Y[:, s] + Z[: ,s] == X + I for s in range(n_scenarios))
        model.addConstr((instance.gozinto.T @ Y_0 + I ==  I_0 ) , name="init_inv") 

        model.update()
        self.Y = Y_0
        self.X = X
        return model
