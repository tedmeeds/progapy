import numpy as np
from progapy.helpers import gamma_rnd, gamma_logprob, gamma_logprob_gradient_free_x
from progapy.priors.prior_distribution import PriorDistribution

class GammaDistribution( PriorDistribution ):
    
  # ========================================== #
  # required implementations by derived classes
  # ========================================== #
  def check_params( self, params ):
    assert params[0] > 0, "alpha aka shape must be > 0"
    assert params[1] > 0, "beta aka scale must be > 0"
    return True
    
  def rand( self, N = 1 ):
    return gamma_rnd( self.p[0], self.p[1], N )
    
  def logdensity( self, x ):
    return np.squeeze( gamma_logprob( x, self.p[0], self.p[1]) )
    
  def logdensity_grad_free_x( self, free_x ):
    #x = np.exp(free_x)
    return gamma_logprob_gradient_free_x( free_x, self.p[0], self.p[1] )
    
  def get_range_of_params( self ):
    D = len(self.p)/2

    L =  0*np.ones( D )
    R =  np.inf*np.ones( D )
    # use sigma as stepsize
    stepsizes =  np.sqrt(self.p[0]/pow(self.p[1],2))*np.ones( D )
    
    return L,R,stepsizes