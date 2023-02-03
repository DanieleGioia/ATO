import time
import gurobipy as grb
from solver.Ato import Ato
from .atoAgent import AtoAgent

class PerfectInfoAgent(AtoAgent):
    def __init__(self, env, solver:Ato , futureDemand):
        super(PerfectInfoAgent, self).__init__(env, solver, futureDemand)
        self.env = env
        self.prb = solver
        self.demand = futureDemand #This class need all the future demand, such that decisions are made with perfect information

        #The agent itself solves a deterministic model to decide the sold and assembled quantity all over the horizon
        sol, model, I, Y, L = self.prb.populate(
            self.env.instance,
            futureDemand
        )
        self.__solve_opt_prb(model)    
        
        #Waring in case of no solution
        if model.status == grb.GRB.Status.OPTIMAL:    
            self.sol = sol.X
            self.Y = Y.X
        else:
            raise ValueError('No optimal solution found')

    def __solve_opt_prb(self, model, time_limit=None, gap=None, verbose=False):
        if gap:
            model.setParam('MIPgap', gap)
        if time_limit:
            model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        if verbose:
            model.setParam('OutputFlag', 1)
            print ('Solving a model with: '+str(model.NumConstrs)+' constraints')
            print ('    and: ' +str(model.NumVars)+ ' variables')
        else:
            model.setParam('OutputFlag', 0)
        start = time.time()
        model.optimize()
        end = time.time()
        return end-start

    def get_action(self, obs):
        return self.sol[:,self.env.current_step], self.Y[:,self.env.current_step]