from .atoAgent import AtoAgent
from solver.solverGurobi import AtoRPMultiStage #Only this solver is allowed due to the seasonalization.
import numpy as np


class MultiStageAgent(AtoAgent):
    def __init__(self, env, solver:AtoRPMultiStage , observedDemand):
        super(MultiStageAgent, self).__init__(env, solver, observedDemand)
        self.env = env
        self.seas = self.env.seas
        self.prb = solver
        self.prb.seasonalize(self.seas)
        self.demand = observedDemand

    #Updated demand according to new observations
    def __updateDemand(self,newObservedDemand):
        self.demand = np.column_stack((self.demand,newObservedDemand))
        return self.demand

    def get_action(self, obs):
        _, sol, _ = self.prb.solve(
            self.env.instance,
            self.demand)
            
        self.prb.updateClock() #notice that it is update AFTER the decision.
        self.__updateDemand(obs['demand'])
        return sol