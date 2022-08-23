# -*- coding: utf-8 -*-
import gurobipy as grb
from solver.solverGurobi.atoG import AtoG


class AtoCVaRProfit(AtoG):
    """   
    max Profit CVar <= r version of the ATO problem
    """
    def __init__(self,**setting):
        super().__init__(**setting)
        self.name = "atoProfitCVaR"
        self.alpha = self.setting["atoProfitCVaR_alpha"]
        self.atoProfitCVaR_limit = self.setting["atoProfitCVaR_limit"]

    def populate(self, instance, scenarios):
        I_0 = instance.inventory
        items = range(instance.n_items)
        machines = range(instance.n_machines)
        n_scenarios = scenarios.shape[1]
        model = grb.Model(self.name)

        zeta = model.addMVar(
            shape=1,
            lb=-grb.GRB.INFINITY,
            ub=grb.GRB.INFINITY,
            name="zeta",
            vtype=grb.GRB.CONTINUOUS,
        )
        Z = model.addMVar(
            shape=n_scenarios,
            vtype=grb.GRB.CONTINUOUS,
            name="Z"
        )
        X = model.addMVar(
            shape=instance.n_components,
            vtype=grb.GRB.CONTINUOUS,
            name='X'
        )
        Y = model.addMVar(
            shape=(instance.n_items, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )

        pi_s = 1.0 / (n_scenarios + 0.0)

        objF = sum(
            pi_s * instance.profits @ Y[:, s]
            for s in range(n_scenarios)
        ) - instance.costs @ X

        model.setObjective(objF, grb.GRB.MAXIMIZE)

        model.addConstrs(Y[j, :] <= scenarios[j, :] for j in items)
        model.addConstrs(instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines)
        model.addConstrs(instance.gozinto.T @ Y[:, s] <= X + I_0 for s in range(n_scenarios))
        model.addConstrs(instance.costs @ X - instance.profits @ Y[:, s] - zeta - Z[s] <= 0 for s in range(n_scenarios))
        expr = zeta + 1.0 / (1.0 - self.alpha) * sum(pi_s * Z[s] for s in range(n_scenarios))
        model.addConstr(expr <= self.atoProfitCVaR_limit)
        
        

        model.update()
        return model
