import numpy as np
from progapy.noise import NoiseModel

class FixedNoiseModel( NoiseModel ):
    
  def check_params(self):
    assert len(self.params) == 1, "fixed model should be vector length 1"
    assert self.params[0] > 0, "param value should be positive"
        
  # def set_prior( self, prior ):  
#     if prior is not None:
#       raise AttributeError( "FixedNoiseModel.prior should be None") 
#     
  def get_nbr_params( self ):
    return 0 # not params because that is the fixed value
      
  # def g_free_params( self, free_params, gp ):
  #   return np.array([]) # empty gradient becase no parameters in this model
  #   
  # def g_params( self, params, gp ):
  #   return np.array([]) # empty gradient becase no parameters in this model
  #   
  def var( self, x = None ):
    if x is None:
      return self.params[0]
      
    self.check_inputs( x )
    
    [N,D] = x.shape
    return self.params[0]*np.eye(N)
    