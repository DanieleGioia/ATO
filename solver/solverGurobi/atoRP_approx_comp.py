# -*- coding: utf-8 -*-
from solver.solverGurobi.atoG import AtoG
import gurobipy as grb
import numpy as np


class AtoRP_approx_comp(AtoG):
    """
    ATO problem with recourse
    SAA methodology
    This applies the first order approximate value of the inventory (FOSVA)

    The setting contains a list (one element per end item) of
    dictionaries with fields:
    -u -> breakpoints
    -v -> slope of each breakpoint

    The number of breakpoints MUST be the same for each item

    For further details please refer to: 
    # "Rolling horizon policies for multi-stage stochastic assemble-to-order problems",
    # by Daniele Giovanni Gioia and Edoardo Fadda and Paolo Brandimarte.
    """
    def __init__(self,**setting):
        super().__init__(**setting)
        self.name = "AtoRP_approx_comp"
        self.fosva_res = self.setting['fosva_res']
        
    def populate(self, instance, scenarios):
        n_scenarios = scenarios.shape[1]
        pi_s = 1.0 / (n_scenarios + 0.0)
        ###
        items = range(instance.n_items)
        components = range(instance.n_components)
        machines = range(instance.n_machines)
        ###
        n_breakpoints = []
        for i in components:
            n_breakpoints.append(len(self.fosva_res[i]['u'])) #checks on the size to be set
        #notice that the breakpoints set must contain 0 as well.
        #
        I_0 = np.array(instance.inventory)
        # model initialisation
        model = grb.Model(self.name)
        # X are first stage solutions, common to every stochastic type of this problem
        X = model.addMVar(
            shape=instance.n_components,
            vtype=grb.GRB.CONTINUOUS,
            name='X'
        )
        #production variable
        Y = model.addMVar(
            shape=(instance.n_items, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )
        #lost sale variable
        L = model.addMVar(
            shape=(instance.n_items, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='L'
        )
        #Z are the components not sold
        Z = model.addMVar(
            shape=(instance.n_components, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='Z'
        )
        #M are the components not sold projected on the 
        # minimum number of items we can build. Here the approximated value is then evaluated.
        M = model.addMVar(
            shape=(instance.n_components, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='M'
        )
        # Piecewise decomposition of M variables
        M_pw_l = []
        for i in components:
            M_pw = model.addMVar(
                shape=(n_breakpoints[i],n_scenarios),
                vtype=grb.GRB.CONTINUOUS,
                name='M_pw_'+str(i)
            )
            M_pw_l.append(M_pw)
        ############ Objective Function
        #sold items
        expr = pi_s * sum(
            instance.profits @ Y[:, s]
            for s in range(n_scenarios)
        )
        #Lost sales
        expr-= pi_s * sum(
            instance.lost_sales @ L[:, s]
            for s in range(n_scenarios)
        )

        #holding costs 
        expr-= pi_s *  sum( (instance.holding_costs*np.ones(instance.n_components) ) @ Z[:, s]
            for s in range(n_scenarios)
        )

        # first stage costs
        expr -= instance.costs[:] @ X

        #Approximated value function
        for i in components:
            expr += pi_s * sum(
                self.fosva_res[i]['v'] @ M_pw_l[i][:,s]
                for s in range(n_scenarios)
            )

        model.setObjective(expr, grb.GRB.MAXIMIZE)

        # Demand constr. and lost sales penalty
        model.addConstrs((Y[j, :] + L[j, :] == scenarios[j, :] for j in items), name="demand_constr")
        # Capacity constraint for each machine
        model.addConstrs((instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines),name="processing_time")
        # End items building
        model.addConstrs((instance.gozinto.T @ Y[:, s] + Z[: ,s] == X + I_0 for s in range(n_scenarios)) , name="end_item_building")        

        # Breakpoints ordering
        for i in components:
            for s in range(n_scenarios):
                model.addConstrs((M_pw_l[i][k,s] <= (self.fosva_res[i]['u'][k+1] - self.fosva_res[i]['u'][k]) for k in range(n_breakpoints[i]-1) ), name="concavixation")

        model.addConstrs( (M[i,:]== sum( M_pw_l[i][k,:] for k in range(n_breakpoints[i]) ) for i in components), name="breakpoints" ) 
        for i in components:
            model.addConstr((M[i,:] <= Z[i,:]) , name="z_to_m")
        
        #updateModel
        model.update()
        return model



class AtoRP_approx_comp_v(AtoG):
    """
    ATO problem with recourse
    SAA methodology
    This optimization allows the computation of the approximate value of the inventory by finite differences
    """
    def __init__(self,**setting):
        super().__init__(**setting)
        self.name = "AtoRP_approx_comp_v"
        
    def populate(self, instance, scenarios):
        n_scenarios = scenarios.shape[1]
        pi_s = 1.0 / (n_scenarios + 0.0)
        ###
        items = range(instance.n_items)
        machines = range(instance.n_machines)
        I_0 = np.array(instance.inventory)
        # model initialisation
        model = grb.Model(self.name)
        # X are first stage solutions, common to every stochastic type of this problem
        X = model.addMVar(
            shape=instance.n_components,
            vtype=grb.GRB.CONTINUOUS,
            name='X'
        )
        #production variable
        Y = model.addMVar(
            shape=(instance.n_items, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )

        #production variable
        Z = model.addMVar(
            shape=(instance.n_components, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )
        #lost sale variable
        L = model.addMVar(
            shape=(instance.n_items, n_scenarios),
            vtype=grb.GRB.CONTINUOUS,
            name='L'
        )

        #sold items
        expr = pi_s * sum(
            instance.profits @ Y[:, s]
            for s in range(n_scenarios)
        )

        #Lost sales
        expr-= pi_s * sum(
            instance.lost_sales @ L[:, s]
            for s in range(n_scenarios)
        )

        # holding costs on the inentory tested for the value
        expr-= pi_s * sum(
            instance.holding_costs*np.ones(instance.n_components) @ Z[:, s]
            for s in range(n_scenarios)
        )

        # first stage costs
        expr -= instance.costs[:] @ X

        model.setObjective(expr, grb.GRB.MAXIMIZE)

        model.addConstrs((Y[j, :] + L[j, :] == scenarios[j, :] for j in items), name="demand_constr") #number of sold items cannot be more than the demand
        model.addConstrs(instance.processing_time[:, m] @ X <= instance.availability[m] for m in machines) # Capacity constraint for each machine
        model.addConstrs(instance.gozinto.T @ Y[:, s] + Z[:, s] == X + I_0 for s in range(n_scenarios))  #components and end items connection
        model.update()
        return model
