from .atoAgent import AtoAgent
from solver.Ato import Ato
import numpy as np


class TwoStageAgent(AtoAgent):
    def __init__(self, env, solver:Ato , observedDemand, full = False):
        super(TwoStageAgent, self).__init__(env, solver, observedDemand)
        self.env = env
        self.prb = solver
        self.demand = observedDemand
        self.full = full

    
    #Update the historical demand according to new observations
    #The available demand is then employed to enhance the aprroximation of the future stages
    def __updateDemand(self,newObservedDemand):
        self.demand = np.column_stack((self.demand,newObservedDemand))
        return self.demand

    def get_action(self, obs):
        # The decision is made by solving the ATO problem.
        # Notice that a two stage agent does not model directly the seasonality, but the available
        # data about the demand is trimmed such that only the of a particular season are taken into account when 
        # the problem is solved.
        # If the full flag is on, the model does not take into account the seasonality, using the entire demand for each step
        if self.full:
            _, solX, solY, _ = self.prb.solve(
                self.env.instance,
                self.demand,
                obs['demand']
            )
        else:
            _, solX, solY, _ = self.prb.solve(
                self.env.instance,
                self.demand[:,np.arange(obs['seasonalFactor'] ,len( self.demand[0,:]) ,self.env.seas)],
                obs['demand']
            )
        self.__updateDemand(obs['demand'])
        return solX, solY







