import numpy as np
from progapy.priors.prior_distribution import PriorDistribution

class EmptyPrior( PriorDistribution ):
   
  def check_params( self ):
    assert self.p is None, " --None-- params for EmptyPrior"
    return True
     
  # ========================================== #
  # required implementations by derived classes
  # ========================================== #
  def rand( self, N = 1 ):
    assert False, "no rand for empty prior"
    
  def logdensity( self, x ):
    return 0.0
    
  def logdensity_grad_free_x( self, free_x ):
    g = np.zeros(self.n)
    return g
    