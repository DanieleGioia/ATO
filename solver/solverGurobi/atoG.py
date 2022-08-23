# -*- coding: utf-8 -*-
import gurobipy as grb
import time
import numpy as np

from solver.Ato import Ato

class AtoG(Ato):
    """
    Interface (super-class) of assembly-to-order solver. Specifications in the subclasses
    """
    def __init__(self,**setting):
        self.name = "atoG"
        self.setting = setting

    def populate(self, instance, scenarios):
        #it initializes the model.
        pass

    def get_solution(self, instance, model, time_limit=None, gap=None, verbose=False):
        if gap:
            model.setParam('MIPgap', gap)
        if time_limit:
            model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        #if verbose:
        #    model.setParam('OutputFlag', 1)
        else:
            model.setParam('OutputFlag', 0)
        model.setParam('LogFile', './logs/gurobi.log')
        # model.write("./logs/model.lp")
        if verbose:
            print ('Solving a model with: '+str(model.NumConstrs)+' constraints')
            print ('    and: ' +str(model.NumVars)+ ' variables')
        start = time.time()
        model.optimize()
        end = time.time()
        comp_time = end - start
        # print("Time to solve {}: {:.2f} [s]".format(self.name, comp_time))
        if model.status == grb.GRB.Status.OPTIMAL:
            n_components = instance.n_components
            sol = [0] * n_components
            # CHECK HOW TO TAKE INFO ABOUT DIMENSION OF VARIABLES FROM THE MODEL
            for i in range(n_components):
                grb_var = model.getVarByName(
                    "X[{}]".format(i)
                )
                sol[i] = grb_var.X
            of = model.getObjective().getValue()
            model.reset()
            return of, sol, comp_time
        else:
            return -1, [], comp_time

    def solve(
        self, instance, scenarios, time_limit=None, gap=None, verbose=False
    ):
        """
        Solve is the method where the problem is stated and then solved.
        :param instance: the dictionary where there is all relevant information for the model building
        :param time_limit: to interrupt the gurobi twoStageSolver with a non optimal solution in case of strict time schedule
        :param gap: if not None, gurobi stops when finding a solution near "gap" to the optimal value
        :param verbose: parameters to be passed to gurobipy package for verbose or not output
        :return: first stage solution in a dict_data['n_components'] array
        """
        model = self.populate(instance, scenarios)
        return self.get_solution(instance, model, time_limit=time_limit, gap=gap, verbose=verbose)

    def change_rhs(self, model, new_set_scenarios):
        model.setAttr(
            "RHS",
            model.getConstrs()[0:np.prod(new_set_scenarios.shape)],
            new_set_scenarios.flatten()
        )
        return model


