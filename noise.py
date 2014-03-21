import numpy as np
import scipy.linalg as spla
from progapy.component import GaussianProcessComponent

class NoiseModel(GaussianProcessComponent):
    
  def check_params(self):
    raise NotImplementedError
    
  def check_inputs( self, x ):
    ndims = len(x.shape) 
    assert ndims == 2, "must be a matrix, even is x is a vector"
      
  def set_params( self, params ):
    self.params      = params
    self.check_params(  )
    self.free_params = np.log( params )
    
  def set_free_params( self, free_params ):
    self.free_params = free_params
    self.params      = np.exp( free_params )
    self.check_params() 
    
  def var( self, X = None ):
    raise NotImplementedError
    
  def std( self, X ):
    return np.sqrt( self.var( X ) )
    
  def f( self, X ):
    return self.var(X)
    
  def loglikelihood_grad_wrt_free_params_using_marginal_typeof_gp( self, gp ):
    g = np.zeros( self.get_nbr_params() )
    J    = self.jacobians( gp.gram, gp.X )
    
    for d in range( self.get_nbr_params() ):
      chol_solve_jacobian = spla.cho_solve((gp.L, True), J[:,:,d] )
      g[d] =   0.5*np.dot( np.dot( gp.Kinv_dot_y.T, J[:,:,d] ), gp.Kinv_dot_y )\
             - 0.5*np.trace( chol_solve_jacobian )

    #g += self.g_free_params_logprior()
    ## g[0]  += (self.prior["signalA"]-1) - self.prior["signalB"]*self.p[0]
    ## g[1:] += -(self.prior["lengthA"]+1) + self.prior["lengthB"]/self.p[1:]

    #print g, g.shape
    if any(np.isnan(g)+np.isinf(g)):
      print "bad grad in kernel"
      pdb.set_trace()
    return g