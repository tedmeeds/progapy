import numpy as np
from progapy.mean import MeanModel

class ConstantMeanModel( MeanModel ):
    
  def check_params(self, params):
    assert len(params) == 1, "a single parameter; later when nbr output>0 ned to change this"
    assert self.prior is None, "no prior for zero mean"
    
  def get_nbr_params( self ):
    return len(self.params) # not params because that is the fixed value
    
  def set_free_params( self, free_params ):
    self.params = free_params.copy()
    
  def set_params( self, params ):
    self.params = params.copy()
  
  def logprior( self ):
    return 0
      
  def g_free_params( self, free_params, gp ):
    return np.array([]) # empty gradient becase no parameters in this model
    
  def g_params( self, params, gp ):
    return np.array([]) # empty gradient becase no parameters in this model
    
  def mu( self, x ):
    N,D = self.check_inputs( x )
    return self.params[0]*np.ones( (N,1) )