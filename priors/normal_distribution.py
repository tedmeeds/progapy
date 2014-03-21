import numpy as np
from progapy.helpers import gaussian_logpdf, gaussian_logpdf_gradient_x
from progapy.priors.prior_distribution import PriorDistribution

class NormalDistribution( PriorDistribution ):
    
  # ========================================== #
  # required implementations by derived classes
  # ========================================== #
  def check_params( self ):
    assert self.p[1] > 0, "variance must be > 0"
    return True
    
  def rand( self, N = 1 ):
    return self.p[0] + np.sqrt(self.p[1])*np.random.randn(N)
    
  def logdensity( self, x ):
    return np.squeeze( gaussian_logpdf( x, self.p[0], np.sqrt(self.p[1])) )
    
  def logdensity_grad_free_x( self, free_x ):
    return gaussian_logpdf_gradient_x( free_x, self.p[0], np.sqrt(self.p[1]) )
    
  def get_range_of_params( self ):
    D = len(self.p)/2

    L = -np.inf*np.ones( D )
    R =  np.inf*np.ones( D )
    stepsizes =  np.sqrt(self.p[1])*np.ones( D )
    
    return L,R,stepsizes
    