import numpy as np
import scipy.linalg as spla

class KernelFunction(object):
  def __init__( self, params, priors = None ):
    self.set_params( params )
    self.priors = priors
    
  def check_inputs( self, x1, x2 ):
    ndims1 = len(x1.shape) 
    assert ndims1 == 2, "must be a matrix, even is x1 is a vector"
    N1,D1 = x1.shape
    N2,D2 = None,None
    if x2 is not None:
      ndims2 = len(x2.shape) 
      N2,D2 = x2.shape
      assert ndims2 == 2, "must be a matrix, even is x2 is a vector"
      assert x1.shape[1] == x2.shape[1], "2nd dim must be the same for both "
      
    return N1,D1,N2,D2
  
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
    return self.params 

  def logprior( self ):
    if self.priors is not None:
      return self.priors.logdensity()
    return 0
  
  def jacobians( self, K, X ):
    raise NotImplementedError
    
  def g_free_params( self, gp, typeof ):
    raise NotImplementedError
    
  def g_params( self, gp, typeof ):
    raise NotImplementedError
    
  def k( self, X1, X2 = None, with_self = False ):
    if X2 is not None:
      return self.k_asym( X1, X2 )
      
    return self.k_sym( X1, with_self )
    
  def k_asym( self, X1, X2 ):
    return self.compute_asymmetric( self.params, X1, X2 )
    
  def k_sym( self, X, with_self = False ):
    return self.compute_symmetric( self.params, X, with_self )

  def g_free_params( self, gp, typeof ):
    
    if typeof == "marginal":
      return self.g_free_params_for_marginal_likelihood( gp )
    elif typeof == "predictive":
      return self.g_free_params_for_predictive_likelihood( gp )
    else:
      assert False, "no other type of gradient"
          
  def g_free_params_for_marginal_likelihood( self, gp ):
    g    = np.zeros( ( gp.N,gp.N,self.get_nbr_params() ) )
    J    = self.jacobians( gp.gram, gp.X )
    
    g = np.zeros( self.get_nbr_params() )
    for d in range( self.get_nbr_params() ):
      chol_solve_jacobian = spla.cho_solve((gp.L, True), J[:,:,d] )
      g[d] =   0.5*np.dot( np.dot( gp.Kinv_dot_y.T, J[:,:,d] ), gp.Kinv_dot_y )\
             - 0.5*np.trace( chol_solve_jacobian )

    ## g[0]  += (self.prior["signalA"]-1) - self.prior["signalB"]*self.p[0]
    ## g[1:] += -(self.prior["lengthA"]+1) + self.prior["lengthB"]/self.p[1:]

    print g, g.shape
    if any(np.isnan(g)+np.isinf(g)):
      print "bad grad in kernel"
      pdb.set_trace()
    return g
    
  