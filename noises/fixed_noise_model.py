import numpy as np
from progapy.noise import NoiseModel

class FixedNoiseModel( NoiseModel ):
    
  def check_params(self, params):
    assert len(params) == 1, "fixed model should be vector length 1"
    assert params[0] > 0, "param value should be positive"
    
    if self.prior is not None:
      if hasattr(self.prior,'g') is False:
        raise AttributeError( "FixedNoiseModel.prior has no method g()")
    
  def get_nbr_params( self ):
    return 0 # not params because that is the fixed value
    
  def set_free_params( self, free_params ):
    self.params = np.exp( free_params )
    self.check_params(self.params)
    
  def set_params( self, params ):
    self.check_params(params)
    self.params = params.copy()
  

  def logprior( self ):
    return 0
      
  def g_free_params( self, free_params, gp ):
    return np.array([]) # empty gradient becase no parameters in this model
    
  def g_params( self, params, gp ):
    return np.array([]) # empty gradient becase no parameters in this model
    
  def var( self, x = None ):
    if x is None:
      return self.params[0]
      
    self.check_inputs( x )
    
    [N,D] = x.shape
    return self.params[0]*np.eye(N)
    