from sampler.Sampler import SamplerIndependent
import numpy as np

class GaussianSampler(SamplerIndependent):
    """
    Normal distributed independent sampler
    """
    def __init__(self, dict_distr: dict):
        super().__init__(dict_distr)
        self.mu = dict_distr["mu"]
        self.sigma = dict_distr["sigma"]
        self.name = "Gaussian_Sampler"

    ##Getter and setter 
    def setMu(self,newMu):
        self.mu = newMu
    def setSigma(self,newSigma):
        self.sigma = newSigma

    #overrided 
    def rescaleAdditive(self,seasonValue):
        self.originalMu = self.mu
        self.setMu(self.mu+seasonValue)
    def rescaleMultiplicative(self,seasonValue):
        self.originalMu = self.mu
        self.setMu(self.mu*seasonValue)
        self.originalSigma = self.sigma
        self.setSigma(self.sigma*seasonValue)

    def revertAdditive(self):
        if not hasattr(self, 'originalMu'):
            pass
        else:
            self.mu = self.originalMu
    def revertMultiplicative(self):
        if not hasattr(self, 'originalSigma'):
            pass
        else:
            self.mu = self.originalMu
            self.sigma =self.originalSigma

    ##Sampling
    def sample(self, n_scenarios: int = 1 ):
        return np.clip(np.random.normal(self.mu,self.sigma,size=(self.nItems,n_scenarios)),
        self.low,self.high)
    #
    def aggregateSampler(self, n_scenarios: int = 1 ):
        return np.clip(np.random.normal(self.mu*self.nItems,self.sigma * (self.nItems**(1/2)),size=(1,n_scenarios)),
        self.low*self.nItems,self.high*self.nItems)

class BiGaussianSampler(GaussianSampler):
    """
    Bi-modal Normal distributed independent sampler
    """ 
    def __init__(self, dict_distr: dict):
        super().__init__(dict_distr)
        self.mu2 = dict_distr["mu2"]
        self.sigma2 = dict_distr["sigma2"]
        self.p = dict_distr['p1']
        self.name = "BiGaussian_Sampler"
        self.rng = np.random.default_rng() #demand shuffler
    
    ##Getter and setter 
    def setMu2(self,newMu):
        self.mu2 = newMu
    def setSigma2(self,newSigma):
        self.sigma2 = newSigma

    #overrided 
    def rescaleAdditive(self,seasonValue):
        super().rescaleAdditive(seasonValue)
        self.originalMu2 = self.mu2
        self.setMu2(self.mu2+seasonValue)
    def rescaleMultiplicative(self,seasonValue):
        super().rescaleMultiplicative(seasonValue)
        self.originalMu2 = self.mu2
        self.setMu2(self.mu2*seasonValue)
        self.originalSigma2 = self.sigma2
        self.setSigma2(self.sigma2*seasonValue)

    def revertAdditive(self):
        if not hasattr(self, 'originalMu2'):
            pass
        else:
            super().revertAdditive()
            self.mu2 = self.originalMu2
    def revertMultiplicative(self):
        if not hasattr(self, 'originalSigma2'):
            pass
        else:
            super().revertMultiplicative()
            self.mu2 = self.originalMu2
            self.sigma2 =self.originalSigma2

    ##Sampling
    def sample(self, n_scenarios: int = 1):
        n_obs_norm_1 = np.random.binomial(n_scenarios, self.p) 
        X1 = super().sample(n_obs_norm_1)
        X2 = np.clip(np.random.normal(self.mu2,self.sigma2,size=(self.nItems,n_scenarios - n_obs_norm_1)),
        self.low,self.high)
        sample = np.concatenate((X1, X2), axis=1)
        self.rng.shuffle(sample,axis=1)
        return sample
    #
    def aggregateSampler(self, n_scenarios: int = 1):
        n_obs_norm_1 = np.random.binomial(n_scenarios, self.p) 
        X1 = super().aggregateSampler(n_obs_norm_1)
        X2 = np.clip(np.random.normal(self.mu2*self.nItems,self.sigma2 * (self.nItems**(1/2)),size=(1,n_scenarios - n_obs_norm_1)),
        self.low*self.nItems,self.high*self.nItems)
        sample = np.concatenate((X1, X2), axis=1)
        self.rng.shuffle(sample,axis=1)
        return sample