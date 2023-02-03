# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as grb
from solver.solverGurobi.atoG_multi import AtoG_multi


class AtoPI(AtoG_multi):
    """
    Version of the ATO where we know in advance the demand and so we can produce optimally. Used for the calculation of
    the EVPI (Expected Value of Perfect Information)
    """
    def __init__(self, **setting):
        super().__init__(**setting)
        self.name = "atoPerfectInfo"

    def populate(self, instance, scenarios):
        n_items, n_time_steps = scenarios.shape
        n_components = instance.n_components
        components = range(n_components)
        items = range(n_items)
        time_steps = range(n_time_steps)
        machines = range(instance.n_machines)
        #initial inventory
        I_0 = np.array(instance.inventory)

        problem_name = "ato_perfect_information"
        model = grb.Model(problem_name)
        # components considered
        X = model.addMVar(
            shape=(instance.n_components, n_time_steps),
            vtype=grb.GRB.CONTINUOUS,
            name='X'
        )
        # sold items
        Y = model.addMVar(
            shape=(instance.n_items, n_time_steps),
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )
        # lost sales per node
        L = model.addMVar(
            shape=(instance.n_items, n_time_steps),
            vtype=grb.GRB.CONTINUOUS,
            name='L'
        )
        #inventory variable
        I = model.addMVar(
            shape=(instance.n_components, n_time_steps),
            vtype=grb.GRB.CONTINUOUS,
            name='I'
        )
        # profits
        expr = sum(
            instance.profits @ Y[:, t]
            for t in time_steps
        )
        # lost sales cost
        expr -= sum(
            instance.lost_sales @ L[:, t]
            for t in time_steps
        )
        # components cost
        expr -= sum(
            instance.costs[:] @ X[:, t]
            for t in time_steps
        )
        # holding costs
        expr -= sum(
            instance.holding_costs[:] @ I[:, t]
            for t in time_steps
        )
        model.setObjective(expr, grb.GRB.MAXIMIZE)

        # Capacity constraint for each machine
        model.addConstrs(instance.processing_time[:, m] @ X[:, t] <= instance.availability[m] for m in machines for t in time_steps)
        
        # Y bounds
        model.addConstrs(
            (Y[j, t] + L[j, t] == scenarios[j, t] for j in items for t in time_steps),
            name="demand_item"
        )

        # Initial condition
        model.addConstr(I[:, 0] == I_0[:] - instance.gozinto.T @ Y[:, 0], name="initial_condition")
        
        # Evolution
        for t in range(1, n_time_steps):
            model.addConstr(I[:, t] + instance.gozinto.T @ Y[:, t] == X[:, t-1] + I[:, t-1], name="evolution")
        model.update()

        #final inv
        model.addConstrs( (I[i,n_time_steps-1] + X[i, n_time_steps-1] >= I_0[i] for i in components) ,name='final_inv')

        return X, model, I, Y, L

    def change_rhs(self, model, new_scenario):
        #no change_rhs implemented for PI
        pass
