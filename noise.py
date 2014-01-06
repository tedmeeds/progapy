import numpy as np

class NoiseModel(object):
  def __init__( self, params, prior = None ):

    self.prior = prior
    self.set_params( params )
    self.check_params( params )
    
  def check_params(self, params):
    raise NotImplementedError
    
  def check_inputs( self, x ):
    ndims = len(x.shape) 
    assert ndims == 2, "must be a matrix, even is x is a vector"
      
  def get_nbr_params( self ):
    raise NotImplementedError
    
  def set_free_params( self, free_params ):
    raise NotImplementedError
    
  def get_free_params( self ):
    raise NotImplementedError
    
  def set_params( self, params ):
    raise NotImplementedError
    
  def get_params( self ):
    raise NotImplementedError
  
  def logprior( self ):
    raise NotImplementedError
      
  def g_free_params( self, gp, typeof ):
    raise NotImplementedError
    
  def g_params( self, gp, typeof ):
    raise NotImplementedError
    
  def var( self, X = None ):
    raise NotImplementedError
    
  def std( self, X ):
    return np.sqrt( self.var( X ) )
    
  def f( self, X ):
    return self.var(X)