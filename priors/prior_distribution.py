import numpy as np

class PriorDistribution( object ):
  
  def __init__( self, params ):
    self.set_params( params )
    
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
    
  def g_logdensity( self, x ):
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