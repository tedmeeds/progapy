class MeanModel(object):
  def __init__( self, params = None, priors = None ):
    self.set_params( params )
    self.priors = priors
    
  def check_inputs( self, x ):
    ndims = len(x.shape) 
    assert ndims == 2, "must be a matrix, even is x is a vector"
    return x.shape 
          
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
  
  def mu( self, X ):
    raise NotImplementedError
    
  def f( self, X ):
    return self.mu(X)
    
  def eval( self, X ):
    return self.mu(X)