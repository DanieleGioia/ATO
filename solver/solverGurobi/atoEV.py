# -*- coding: utf-8 -*-
import gurobipy as grb
from solver.solverGurobi.atoG import AtoG


class AtoEV(AtoG):
    """
    Single stage ATO problem that maximizes the expected net profit of the problem without a recourse function
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
        #sold items ON AVERAGE
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

        model.addConstr(Y <= scenarios.mean(1)) #number of sold items cannot be more than the AVERAGE demand
        model.addConstrs(instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines) #machine availability constraint
        model.addConstr(instance.gozinto.T @ Y <= X) #components and end items connection
        model.update()
        return model
