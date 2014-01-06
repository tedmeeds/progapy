import numpy as np
from progapy.mean import MeanModel

class ZeroMeanModel( MeanModel ):
    
  def check_params(self, params):
    assert params is None, "there are no parameters for zero mean model"
    assert self.prior is None, "no prior for zero mean"
    
  def get_nbr_params( self ):
    return 0 # not params because that is the fixed value
    
  def set_free_params( self, free_params ):
    pass # do nothing
    
  def set_params( self, params ):
    pass # do nothing
  
  def logprior( self ):
    return 0
      
  def g_free_params( self, free_params, gp ):
    return np.array([]) # empty gradient becase no parameters in this model
    
  def g_params( self, params, gp ):
    return np.array([]) # empty gradient becase no parameters in this model
    
  def mu( self, x ):
    N,D = self.check_inputs( x )
    return np.zeros( (N,1) )