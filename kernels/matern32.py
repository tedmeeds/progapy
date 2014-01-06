import numpy as np
import scipy.linalg as spla
from progapy.helpers import fast_distance, fast_grad_distance
from progapy.kernel import KernelFunction
import pdb

class Matern32Function( KernelFunction ):
  
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
      
  def compute_symmetric( self, params, X, with_self ):
    N,D,N2,D2 = self.check_inputs( X, X )
    
    if with_self:
      return params[0]*np.ones( (N,1) )
    else:
      d = np.sqrt(3)*np.sqrt( fast_distance( params[1:], X ) )
      return params[0]*(1.0 + d)*np.exp( -d )  

  def compute_asymmetric( self, params, X1, X2 ):
    N1,D1,N2,D2 = self.check_inputs( X1, X2 )
    
    return params[0]*np.exp( -0.5*fast_distance( self.params[1:], X1, X2 ) ) 
    
  def g_free_params( self, gp, typeof ):
    
    if typeof == "marginal":
      return self.g_free_params_for_marginal_likelihood( gp )
    elif typeof == "predictive":
      return self.g_free_params_for_predictive_likelihood( gp )
    else:
      assert False, "no other type of gradient"
      
  # assumes free parameters...
  def jacobians( self, K, X ):
    N = len(X)
    g    = np.zeros( ( N,N,self.get_nbr_params() ) )
    g[:,:,0] = K
    
    for i in range(N):
      for j in range(N):
        r = np.sqrt(3)*np.abs( X[i,:] - X[j,:] )/self.params[1:]
        for d in range(self.get_nbr_params()-1):
          g[i,j,d+1] += self.params[0]*r[d]*r[d]*np.exp( -r.sum() ) #/(self.params[d+1])
      
    return g
    

    