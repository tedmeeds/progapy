import numpy as np

class PriorDistribution( object ):
  
  def __init__( self, params, ids = np.array([0])):
    self.n = len(ids)
    self.set_ids( ids )
    self.set_params( params )
    
  def set_ids( self, ids ):
    self.ids = ids
    
  def set_params( self, params ):
    assert self.check_params( params ), "error with params"
    self.p = params
    
  # ========================================== #
  # required implementations by derived classes
  # ========================================== #
  def check_params( self ):
    raise NotImplementedError
    
  # def check_input( self, x ):
  #   raise NotImplementedError
    
  def rand( self, N = 1 ):
    raise NotImplementedError
    
  def logdensity( self, x ):
    raise NotImplementedError
    
  def logdensity_grad_free_x( self, free_x ):
    raise NotImplementedError
    
  # ========================================== #
  # default implementations
  # ========================================== #    
  def logpdf( self, x ):
    return self.logdensity(x)
    
  def lpdf( self, x ):
    return self.logdensity(x)
    
  def pdf( self, x ):
    return np.exp( self.logdensity(x) )
    
  def g_logpdf( self, x ):
    return self.g_logdensity( x )
    
  def g_lpdf( self, x ):
    return self.g_logdensity( x )