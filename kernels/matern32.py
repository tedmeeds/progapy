import numpy as np
import scipy.linalg as spla
from progapy.helpers import fast_distance, fast_grad_distance
from progapy.kernel import KernelFunction
import pdb

class Matern32Function( KernelFunction ):
  
  def shrink_length_scales(self, factor ):
    self.params[1:] *= factor
        
  def compute_symmetric( self, params, X, with_self ):
    N,D,N2,D2 = self.check_inputs( X, X )
    
    if with_self:
      return params[0]*np.ones( (N,1) )
    else:
      d = np.sqrt(3)*np.sqrt( fast_distance( params[1:], X ) )
      return params[0]*(1.0 + d)*np.exp( -d )  

  def compute_asymmetric( self, params, X1, X2 ):
    N1,D1,N2,D2 = self.check_inputs( X1, X2 )
    d = np.sqrt(3)*np.sqrt( fast_distance( self.params[1:], X1, X2 ) )
    return params[0]*(1.0 + d)*np.exp( -d )  
    
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
    
  def get_range_of_params( self ):
    L =  0*np.ones( self.get_nbr_params() )
    R =  np.inf*np.ones( self.get_nbr_params() )
    stepsizes =  np.ones( self.get_nbr_params() )
    
    if self.prior is not None:
      return self.prior.get_range_of_params()
    else:
      return L,R,stepsizes
    

    