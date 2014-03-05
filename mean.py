import numpy as np
import scipy.linalg as spla
import pdb
from progapy.component import GaussianProcessComponent

class MeanModel(GaussianProcessComponent):
  
  def check_inputs( self, x ):
    ndims = len(x.shape) 
    assert ndims == 2, "must be a matrix, even if x is a vector"
    return x.shape 
          
  def set_params( self, params ):
    self.params      = params
    self.free_params = params # params are free
    
  def set_free_params( self, free_params ):
    self.free_params = free_params
    self.params      = free_params
  
  def mu( self, X ):
    raise NotImplementedError
    
  def f( self, X ):
    return self.mu(X)
    
  def eval( self, X ):
    return self.mu(X)
          
  def loglikelihood_grad_wrt_free_params_using_marginal_typeof_gp( self, gp ):
    g = np.zeros( self.get_nbr_params() )
    grads    = self.grads( gp.X )
    #pdb.set_trace()
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