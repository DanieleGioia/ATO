from sampler import *
import numpy as np

class MultiStageSampler(Sampler):
    """
    MultiStage sampler. It adapts the other samplers to a multistage setting.
    It is possible to set seasonality. The number of scenarios is now the horizon.
    params:
    -multiplicativeSeas: array of %_variation with seasonality length e.g. [1,1.4,0.6]
        ->+0% step one, +40% step two, -40% step three, repeat
        its deafult value must be [1]
    -additiveSeas: array of additive variation with seasonality length e.g. [50,-30,-20]
        ->+50 step one, -30 step two, -20 step three, repeat
        its deafult value must be [0]
    Although it is not recommended, these two effects can be mixed
    """    
    def __init__(self, dict_distr: dict, sampler: Sampler):
        self.setInstanceOptions(dict_distr)
        self.name = 'MultiStage Sampler'
        self.sampler = sampler
        #Seasonality
        self.additiveSeas = dict_distr["additiveSeas"]
        self.multiplicativeSeas = dict_distr["multiplicativeSeas"]
        #The seasonality must be normalized such that it does not vary the expected value.
        if sum(self.additiveSeas)!= 0:
            raise ValueError('The additive seasonality must sum to 0, oth it exists an equivalent formulation that sums to zero with a different expected value')
        if np.round(sum(self.multiplicativeSeas))!= len(self.multiplicativeSeas):
            raise ValueError('The multiplicative seasonality must sum to the array length, oth it exists an equivalent formulation that sums to the length of the array with a different expected value and std')

    def __addMultiplicativeSeas(self,seasonValue):
        self.sampler.revertMultiplicative()
        self.sampler.rescaleMultiplicative(seasonValue)

    def __addAdditiveSeas(self,seasonValue):
        self.sampler.revertAdditive()
        self.sampler.rescaleAdditive(seasonValue)

    def sample(self, horizon: int = 1):
        seasAdd = (len(self.additiveSeas) != 1) #additive seasonality?
        if seasAdd:
            q, r = divmod(horizon, len(self.additiveSeas)) #how many reps
            additiveValues = q * self.additiveSeas + self.additiveSeas[:r] #reps
        seasMul = (len(self.multiplicativeSeas) != 1 ) #multiplicative seasonality?
        if seasMul:
            q, r = divmod(horizon, len(self.multiplicativeSeas)) #how many reps
            multiplicativeValues = q * self.multiplicativeSeas + self.multiplicativeSeas[:r] #reps
        #prealloc            
        smpl = np.zeros((self.nItems,horizon))
        for i in range(horizon):
            if seasAdd:
                self.__addAdditiveSeas(additiveValues[i])
            if seasMul:
                self.__addMultiplicativeSeas(multiplicativeValues[i])
            smpl[:,i] = self.sampler.sample()[:,0]
        return smpl
