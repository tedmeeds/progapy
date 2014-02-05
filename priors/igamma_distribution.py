import numpy as np
from progapy.helpers import invgamma_rnd, invgamma_logprob, invgamma_logprob_gradient
from progapy.priors.prior_distribution import PriorDistribution

class InverseGammaDistribution( PriorDistribution ):
    
  # ========================================== #
  # required implementations by derived classes
  # ========================================== #
  def check_params( self, params ):
    assert params[0] > 0, "alpha aka shape must be > 0"
    assert params[1] > 0, "beta aka scale must be > 0"
    return True
  # def check_input( self, x ):
  #   raise NotImplementedError
    
  def rand( self, N = 1 ):
    return invgamma_rnd( self.p[0], self.p[1], N )
    
  def logdensity( self, x ):
    return np.squeeze( invgamma_logprob( x, self.p[0], self.p[1]) )
    
  def g_logdensity( self, x ):
    #print "computing gradient for x = %f a = %f b = %f"%(x, self.p[0], self.p[1])
    return invgamma_logprob_gradient( x, self.p[0], self.p[1] )
    