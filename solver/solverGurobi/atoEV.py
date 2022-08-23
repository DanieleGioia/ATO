# -*- coding: utf-8 -*-
import gurobipy as grb
from solver.solverGurobi.atoG import AtoG


class AtoEV(AtoG):
    """
    Standard version of the ATO problem, see the companion paper/thesis for the explicit model
    """
    def __init__(self,**setting):        
        super().__init__(**setting)
        self.name = "EV"

    def populate(self, instance, scenarios):
        machines = range(instance.n_machines)
        # model initialisation
        model = grb.Model(self.name)
        # X are first stage solutions, common to every stochastic type of this problem
        X = model.addMVar(
            shape=instance.n_components,
            vtype=grb.GRB.CONTINUOUS,
            name='X'
        )

        Y = model.addMVar(
            shape=instance.n_items,
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
            # ub=demand
        )
        # revenues coming from expected sales
        expr = instance.profits @ Y
        # first stage costs
        expr -= instance.costs @ X
        model.setObjective(expr, grb.GRB.MAXIMIZE)

        # Capacity constraint for each machine
        model.addConstr(Y <= scenarios.mean(1))
        model.addConstrs(instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines)
        model.addConstr(instance.gozinto.T @ Y <= X)
        model.update()
        return model
