class KernelFunction(object):
  def __init__( self, params, priors = None ):
    self.params = params
    self.priors = priors
    
  def check_inputs( self, x1, x2 ):
    ndims1 = len(x1.shape) 
    assert ndims1 == 2, "must be a matrix, even is x1 is a vector"
    if x2 is not None:
      ndims2 = len(x2.shape) 
      assert ndims2 == 2, "must be a matrix, even is x2 is a vector"
      assert x1.shape[1] == x2.shape[1], "2nd dim must be the same for both "
      
  def get_nbr_params( self ):
    raise NotImplementedError
    
  def set_free_params( self, free_params ):
    raise NotImplementedError
    
  def set_params( self, free_params ):
    raise NotImplementedError
    
  def g_free_params( self, free_params, x1, x2 ):
    raise NotImplementedError
    
  def g_params( self, params, x1, x2 ):
    raise NotImplementedError
  
  def compute( self, params, x1, x2 = None ):
    raise NotImplementedError
    
  