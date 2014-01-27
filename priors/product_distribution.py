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
  def check_params( self, x ):
    raise NotImplementedError
    
  def check_input( self, x ):
    raise NotImplementedError
    
  def rand( self, N = 1 ):
    raise NotImplementedError
    
  def logdensity( self, x ):
    raise NotImplementedError
    
  def g_logdensity( self, x ):
    raise NotImplementedError
    