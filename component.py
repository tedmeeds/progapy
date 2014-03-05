import numpy as np
import scipy.linalg as spla
import pdb

# a base class for mean, kernel, noise models

class GaussianProcessComponent(object):
  def __init__( self, params = None, prior = None ):
    self.set_params( params )
    self.set_prior( prior )
       
  def set_params( self, params ):
    raise NotImplementedError
    
  def set_prior( self, prior ):
    self.prior = prior 
    
  def set_free_params( self, free_params ):
    raise NotImplementedError
    
  def set_p_or_fp( self, p = None, fp = None ):
    if fp is not None:
      assert len(fp) == self.get_nbr_params(), "incorrect fp given"
      self.set_free_params( fp )
    elif p is not None:
      assert len(p) == self.get_nbr_params(), "incorrect p given"
      self.set_params( p )  
    
  def get_nbr_params( self ):
    return len(self.params)
  
  def get_free_params( self ):
    return self.free_params
    
  def get_params( self ):
    return self.params 
    
  def get_range_of_params( self ):
    raise NotImplementedError
    
  def logprior( self ):
    if self.prior is not None:
      return self.prior.logdensity( self.params )
    return 0.0
  
  def logprior_grad_wrt_free_params( self ):
    if self.prior is not None:
      return self.prior.logdensity_grad_free_x( self.free_params )
    else:
      return np.zeros( (self.get_nbr_params()))
    
  def loglikelihood_grad_wrt_free_params( self, gp ):
    # marginal is typical GP
    if gp.typeof == "marginal":
      return self.loglikelihood_grad_wrt_free_params_using_marginal_typeof_gp( gp )
      
    # predictive is sparse GP (likelihood is prediction of sparse pseudodata)
    elif gp.typeof == "predictive":
      return self.loglikelihood_grad_wrt_free_params_using_predictive_typeof_gp( gp )
    else:
      assert False, "no other type of gradient"
          
  def loglikelihood_grad_wrt_free_params_using_marginal_typeof_gp( self, gp ):
    raise NotImplementedError
    
  def loglikelihood_grad_wrt_free_params_using_predictive_typeof_gp( self, gp ):
    raise NotImplementedError
    
  # def loglikelihood_grad_wrt_free_params_using_marginal_typeof_gp( self, gp ):
  #   grads    = self.grads( gp.X )
  #   pdb.set_trace()
  #   g = np.zeros( self.get_nbr_params() )
  #   for d in range( self.get_nbr_params() ):
  #     chol_solve_jacobian = spla.cho_solve((gp.L, True), grads[:,d] )
  #     g[d] =   np.dot( gp.Kinv_dot_y.T, grads[:,d] )
  # 
  #   ## g[0]  += (self.prior["signalA"]-1) - self.prior["signalB"]*self.p[0]
  #   ## g[1:] += -(self.prior["lengthA"]+1) + self.prior["lengthB"]/self.p[1:]
  # 
  #   print g, g.shape
  #   if any(np.isnan(g)+np.isinf(g)):
  #     print "bad grad in kernel"
  #     pdb.set_trace()
  #   return g