import numpy as np
import scipy.linalg as spla

class MeanModel(object):
  def __init__( self, params = None, priors = None ):
    self.set_params( params )
    self.priors = priors
    
  def check_inputs( self, x ):
    ndims = len(x.shape) 
    assert ndims == 2, "must be a matrix, even is x is a vector"
    return x.shape 
          
  def set_params( self, params ):
    self.params      = params
    self.free_params = params # params are free
    
  def set_free_params( self, free_params ):
    self.free_params = free_params
    self.params      = free_params
    
  def get_nbr_params( self ):
    return len(self.params)
  
  def get_free_params( self ):
    return self.free_params
    
  def get_params( self ):
    return self.params 

  def logprior( self ):
    if self.priors is not None:
      return self.priors.logdensity( self.params )
    return 0
    
  def g_params( self, gp, typeof ):
    raise NotImplementedError
  
  def mu( self, X ):
    raise NotImplementedError
    
  def f( self, X ):
    return self.mu(X)
    
  def eval( self, X ):
    return self.mu(X)
    
  def g_free_params( self, gp, typeof ):
    
    if typeof == "marginal":
      return self.g_free_params_for_marginal_likelihood( gp )
    elif typeof == "predictive":
      return self.g_free_params_for_predictive_likelihood( gp )
    else:
      assert False, "no other type of gradient"
          
  def g_free_params_for_marginal_likelihood( self, gp ):
    grads    = self.grads( gp.X )
    g = np.zeros( self.get_nbr_params() )
    for d in range( self.get_nbr_params() ):
      chol_solve_jacobian = spla.cho_solve((gp.L, True), grads[:,d] )
      g[d] =   np.dot( gp.Kinv_dot_y.T, grads[:,d] )

    ## g[0]  += (self.prior["signalA"]-1) - self.prior["signalB"]*self.p[0]
    ## g[1:] += -(self.prior["lengthA"]+1) + self.prior["lengthB"]/self.p[1:]

    print g, g.shape
    if any(np.isnan(g)+np.isinf(g)):
      print "bad grad in kernel"
      pdb.set_trace()
    return g