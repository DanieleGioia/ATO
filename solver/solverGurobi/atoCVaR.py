# -*- coding: utf-8 -*-
import gurobipy as grb
from solver.solverGurobi.atoG import AtoG


class AtoCVaR(AtoG):
    """   
    CVar version of the ATO problem
    @@@Currently, it does not support the inventory@@@
    """
    def __init__(self,**setting):
        super().__init__(**setting)
        self.name = "ato_CVaR"
        #level of the c value at risk
        self.alpha = self.setting["CVaR_alpha"]
        #minimum expected profit
        self.expected_profit = self.setting["CVaR_expected_profit"]

    def populate(self, instance, scenarios):
        items = range(instance.n_items)
        machines = range(instance.n_machines)
        n_scenarios = scenarios.shape[1]
        model = grb.Model(self.name)
        #variables of the value at risk formulation
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
        #production decision: number of components
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
        #probability of a scenario
        pi_s = 1.0 / (n_scenarios + 0.0)
        objF = 1.0 / (1.0 - self.alpha) * pi_s * sum(Z[s] for s in range(n_scenarios)) #CV@R objective function formulation
        model.setObjective(zeta + objF, grb.GRB.MINIMIZE)

        model.addConstrs(Y[j, :] <= scenarios[j, :] for j in items) #number of sold items cannot be more than the demand
        model.addConstrs(instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines) #machine availability constraint
        model.addConstrs(instance.gozinto.T @ Y[:, s] <= X for s in range(n_scenarios)) #components and end items connection
        model.addConstrs(instance.costs @ X - instance.profits @ Y[:, s] - zeta - Z[s] <= 0 for s in range(n_scenarios)) #CV@R formulation

        expr = pi_s * sum(
            instance.profits @ Y[:, s] #second stage  profits 
            for s in range(n_scenarios)
        ) - instance.costs @ X # first stage costs

        #constraint on the minimum expected net profit
        model.addConstr(
            expr >= self.expected_profit,
            "mean_earning"
        )

        model.update()
        return model
