import numpy as np
from progapy.noise import NoiseModel

class StandardNoiseModel( NoiseModel ):
    
  def check_params(self, params):
    assert len(params) == 1, "fixed model should be vector length 1"
    assert params[0] > 0, "param value should be positive"
    
    if self.priors is not None:
      if hasattr(self.priors,'g') is False:
        raise AttributeError( "StandardNoiseModel.prior has no method g()")
        
  def var( self, x = None ):
    if x is None:
      return self.params[0]
    
    self.check_inputs( x )
  
    [N,D] = x.shape
    return self.params[0]*np.eye(N)
         
  # def g_free_params( self, free_params, gp ):
  #   if self.priors is None:
  #     return self.params*gp.g_noise()
  #   else:
  #     return self.params*gp.g_noise() + self.params*self.priors.g()
  #   
  # def g_params( self, params, gp ):
  #   if self.priors is None:
  #     return gp.g_noise()
  #   else:
  #     return gp.g_noise() + self.priors.g()
      
  # assumes free parameters...
  def jacobians( self, K, X ):
    N = len(X)
    D = self.get_nbr_params()
    J = np.zeros( (N,N,D) )
    J[:,:,0] = self.params[0]*np.eye(len(X))
    return J