from sampler.Sampler import SamplerDependencies
from sampler.Gaussian_Sampl import GaussianSampler
import numpy as np

class HierarchicalSampler(SamplerDependencies):
    """
    Hierarchical distribution with one distribution set to sample the single family value
    and a Dirichlet distribution to divide the demand per family within its items.

    The class require a Gaussian Sampler class input for the families and the parameters are assumed to be set 
    on the dictionary

    The class requires the gozinto settings to correlate the families
    """
    def __init__(self, dict_distr: dict , dict_gozinto: dict , familyDistr: GaussianSampler):
        """
        Setting contains a dictionary with the parameters
        """
        super().__init__(dict_distr,dict_gozinto)
        self.name = "Hierarchical_Sampler"
        self.famSampler = familyDistr
        inFamCann = dict_distr['cannibalization'] # must be int and ge than 1
        #Inner distributions per family generation
        self.dirichParams = []
        for num_fam in self.itemsPerFamily:
            self.dirichParams.append(np.random.choice(inFamCann*num_fam,num_fam)+1)
    #Setter e getter
    def getDirichFamily(self,familyIndx:int):
        return self.dirichParams[familyIndx]
    def setDirichFamily(self,familyIndx:int,values:np.array):
        if self.dirichParams[familyIndx].size != values.size :
            raise ValueError('family size error on dirich distribution')
        self.dirichParams[familyIndx] = values

    #Variation on the inner distributions of the families
    def shuffleInnerFamiliesShare(self,perc = 0.20,verbose = False):
        if verbose:
            print('Old inner share per family')
            for fam in range(len(self.dirichParams)):
                print((self.dirichParams[fam]/sum(self.dirichParams[fam]))*100)
        #variation
        for fam in range(len(self.dirichParams)):
            self.dirichParams[fam] = self.dirichParams[fam] + self.dirichParams[fam]*np.random.uniform(-perc,perc,len(self.dirichParams[fam]))
        if verbose:
            print('New inner share per family')
            for fam in range(len(self.dirichParams)):
                print((self.dirichParams[fam]/sum(self.dirichParams[fam]))*100)




    #overrided 
    def rescaleAdditive(self,seasonValue):
        self.famSampler.rescaleAdditive(seasonValue)
    def rescaleMultiplicative(self,seasonValue):
        self.famSampler.rescaleMultiplicative(seasonValue)

    def revertAdditive(self):
        self.famSampler.revertAdditive()
    def revertMultiplicative(self):
        self.famSampler.revertMultiplicative()


    #Sampling
    def sample(self, n_scenarios: int = 1):
        #Pre-allocation
        smpl = np.zeros((self.nItems,n_scenarios))
        # OUTCAST 
        self.famSampler.setNItems(self.outcastItems)
        start_out = self.nItems - self.outcastItems
        end_out = self.nItems
        smpl[start_out:end_out,:] = self.famSampler.sample(n_scenarios)
        # ITEMS IN FAMILY
        start_row = 0
        for i,num_fam in enumerate(self.itemsPerFamily):
            self.famSampler.setNItems(num_fam)
            totalDemandFamily = self.famSampler.aggregateSampler(n_scenarios)
            end_row = start_row + num_fam
            idioValues = np.random.dirichlet(self.dirichParams[i],size=n_scenarios).T
            smpl[start_row:end_row,:] = np.clip( idioValues*totalDemandFamily,self.low,self.high)
            start_row = end_row
        return smpl