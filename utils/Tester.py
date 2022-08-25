# -*- coding: utf-8 -*-
from solver.Ato import Ato
from sampler.Sampler import Sampler
from instances import *
import numpy as np
from tqdm import tqdm
import gurobipy as grb


class Tester():
    """
    """
    def __init__(self):
        pass

    def compare_sol(
        self, inst: Instance,  solution: dict, n_scenarios: int, sampler: Sampler, benchmark: Ato=None):
        """
        :param inst: is the instance used in both solving and evaluation (in eval change in the demand)
        :param sols: is a dictionary of solutions, labelled according to the array of solution's type (RP, LDR and so on)
        :param n_scenarios: is the number of scenarios with which to evaluate all the solutions
        :param sampler: an object of class Sampler to generate the demand
        :param benchmark: an object of class ATO to generate a benchmark solution, solved with the sampled demand
        """
        profit_ans_dict = {}
        # off sample demands for evaluation purposes
        demands = sampler.sample(
            n_scenarios=n_scenarios
        )
        for i in solution.keys():
            profit_raw_data = self.simulate_productions(
                inst, solution[i], demands, benchmark=benchmark
            )
            profit_ans_dict[i] = profit_raw_data

        return profit_ans_dict            

    def simulate_productions(
        self, inst: Instance, sol: np.array, demands: np.array, benchmark: Ato=None
    ):
    
        n_scenarios = demands.shape[1]
        profit_ans = np.zeros(n_scenarios)
        obj, _ = self.produce(inst, sol, demands[:, 0])
        profit_ans[0] = obj
        if benchmark:
            profit_bench = np.zeros(n_scenarios)
            profit_bench[0], _, _ = benchmark.solve(inst, demands[:, 0])

        for s in tqdm(range(1, n_scenarios)):
            self.model.setAttr(
                "RHS",
                self.model.getConstrs()[0:inst.n_items], demands[:, s]
            )
            self.model.optimize()
            obj = self.model.getObjective().getValue()
            profit_ans[s] = obj
            if benchmark:
                profit_bench[s], _, _ = benchmark.solve(inst, demands[:, s])
        if benchmark:
            return (profit_bench - profit_ans) / profit_bench
        else:
            return profit_ans

    def produce(
        self, inst, sol: np.array, demands: np.array
    ):
        components = range(inst.n_components)
        items = range(inst.n_items)
        I_0 = inst.inventory
        cost_sol = 0
        for i in range(len(sol)):
            cost_sol += (inst.costs[i] + 0.0) * (sol[i] + 0.0)

        # we create second stage model, with fixed first stage solution
        self.model = grb.Model("ato_2nd_stage")
        # only variable is what to assembly, giving fixed number of components and perfect demand information
        Y = self.model.addVars(
            inst.n_items,
            vtype=grb.GRB.CONTINUOUS,
            name='Y'
        )
        # first scenario problem
        s = 0
        expr = grb.quicksum(
            inst.profits[j] * Y[j]
            for j in items
        )
        expr -= grb.quicksum(
            inst.costs[i] * sol[i] for i in components
        )
        self.model.setObjective(expr, grb.GRB.MAXIMIZE)

        for j in items:
            self.model.addConstr(
                Y[j] <= demands[j],
                "demand_item_{}".format(j)
            )
        for i in components:
            expr = grb.quicksum(
                inst.gozinto[j, i] * Y[j] for j in items
            )
            self.model.addConstr(
                expr <= sol[i] + I_0[i],
                "gozinto_item_{}".format(i)
            )

        self.model.update()
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        obj = self.model.getObjective().getValue()
        
        return obj, np.array([Y[j].X for j in items])
