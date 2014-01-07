import numpy as np
import scipy.linalg as spla

class NoiseModel(object):
  def __init__( self, params, priors = None ):

    self.priors = priors
    self.set_params( params )
    self.check_params( params )
    
  def check_params(self, params):
    raise NotImplementedError
    
  def check_inputs( self, x ):
    ndims = len(x.shape) 
    assert ndims == 2, "must be a matrix, even is x is a vector"
      
  def set_params( self, params ):
    self.params      = params
    self.free_params = np.log( params )
    
  def set_free_params( self, free_params ):
    self.free_params = free_params
    self.params      = np.exp( free_params )
    
  def get_nbr_params( self ):
    return len(self.params)
  
  def get_free_params( self ):
    return self.free_params
    
  def get_params( self ):
    return params 

  def logprior( self ):
    if self.priors is not None:
      return self.priors.logdensity()
    return 0
    
  def g_params( self, gp, typeof ):
    raise NotImplementedError
    
  def var( self, X = None ):
    raise NotImplementedError
    
  def std( self, X ):
    return np.sqrt( self.var( X ) )
    
  def f( self, X ):
    return self.var(X)
    
  def g_free_params( self, gp, typeof ):
    
    if typeof == "marginal":
      return self.g_free_params_for_marginal_likelihood( gp )
    elif typeof == "predictive":
      return self.g_free_params_for_predictive_likelihood( gp )
    else:
      assert False, "no other type of gradient"
      
  def g_free_params_for_marginal_likelihood( self, gp ):
    pass
    
  def g_free_params_for_marginal_likelihood( self, gp ):
    g    = np.zeros( ( gp.N,gp.N,self.get_nbr_params() ) )
    J    = self.jacobians( gp.gram, gp.X )
    
    g = np.zeros( self.get_nbr_params() )
    for d in range( self.get_nbr_params() ):
      chol_solve_jacobian = spla.cho_solve((gp.L, True), J[:,:,d] )
      g[d] =   0.5*np.dot( np.dot( gp.chol_solve_y.T, J[:,:,d] ), gp.chol_solve_y )\
             - 0.5*np.trace( chol_solve_jacobian )

    ## g[0]  += (self.prior["signalA"]-1) - self.prior["signalB"]*self.p[0]
    ## g[1:] += -(self.prior["lengthA"]+1) + self.prior["lengthB"]/self.p[1:]

    print g, g.shape
    if any(np.isnan(g)+np.isinf(g)):
      print "bad grad in kernel"
      pdb.set_trace()
    return g