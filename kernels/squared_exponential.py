import numpy as np
import scipy.linalg as spla
from progapy.helpers import fast_distance, fast_grad_distance
from progapy.kernel import KernelFunction
import pdb

class SquaredExponentialFunction( KernelFunction ):
       
  def compute_symmetric( self, params, X, with_self ):
    N,D,N2,D2 = self.check_inputs( X, X )
    
    if with_self:
      return params[0]*np.ones( (N,1) )
    else:
      return params[0]*np.exp( -0.5*fast_distance( params[1:], X ) )  

  def compute_asymmetric( self, params, X1, X2 ):
    N1,D1,N2,D2 = self.check_inputs( X1, X2 )
    
    return params[0]*np.exp( -0.5*fast_distance( self.params[1:], X1, X2 ) ) 
      
  # assumes free parameters...
  def jacobians( self, K, X ):
    N = len(X)
    g    = np.zeros( ( N,N,self.get_nbr_params() ) )
    g[:,:,0] = K
    
    for i in range(N):
      for j in range(N):
        dif =( X[i,:] - X[j,:])**2
        for d in range(self.get_nbr_params()-1):
          g[i,j,d+1] += K[i,j]*dif[d]/(self.params[d+1]**2)
      
    return g
    
    