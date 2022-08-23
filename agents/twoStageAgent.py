from .atoAgent import AtoAgent
from solver.Ato import Ato
import numpy as np


class TwoStageAgent(AtoAgent):
    def __init__(self, env, solver:Ato , observedDemand):
        super(TwoStageAgent, self).__init__(env, solver, observedDemand)
        self.env = env
        self.prb = solver
        self.demand = observedDemand

    #Updated demand according to new observations
    def __updateDemand(self,newObservedDemand):
        self.demand = np.column_stack((self.demand,newObservedDemand))
        return self.demand

    def get_action(self, obs):
        _, sol, _ = self.prb.solve(
            self.env.instance,
            self.demand[:,np.arange(obs['seasonalFactor'] ,len( self.demand[0,:]) ,self.env.seas)]
        )
        self.__updateDemand(obs['demand'])
        return sol







