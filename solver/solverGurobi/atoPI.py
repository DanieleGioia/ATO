# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as grb
from solver.solverGurobi.atoG import AtoG


class AtoPI(AtoG):
    """
    Version of the ATO where we know in advance the demand and so we can produce optimally. Used for the calculation of
    the EVPI (Expected Value of Perfect Information)
    """
    def __init__(self, **setting):
        super().__init__(**setting)
        self.name = "atoPerfectInfo"

    def populate(self, instance, scenarios):
        n_components = instance.n_components
        components = range(n_components)
        n_items = instance.n_items
        items = range(n_items)
        machines = range(instance.n_machines)
        problem_name = "ato_ws"
        model = grb.Model(problem_name)
        X = model.addMVar(
            shape=instance.n_components,
            vtype=grb.GRB.CONTINUOUS,
            name='X'
        )
        Y = model.addMVar(
            shape=instance.n_items,
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )
        expr = instance.profits @ Y - instance.costs @ X
        
        model.setObjective(expr, grb.GRB.MAXIMIZE)
        
        model.addConstr(Y <= scenarios, name="demand_constr")
        model.addConstrs(instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines)
        model.addConstr(instance.gozinto.T @ Y <= X)
        model.update()
        return model

    def change_rhs(self, model, new_scenario):
        for j in range(len(new_scenario)):
            model.setAttr(
                "RHS",
                model.getConstrByName(f"demand_item_{j}"),
                new_scenario[j]
            )
        return model
