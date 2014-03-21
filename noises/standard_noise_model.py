import numpy as np
from progapy.noise import NoiseModel

class StandardNoiseModel( NoiseModel ):
    
  def check_params(self ):
    assert len(self.params) == 1, "fixed model should be vector length 1"
    assert self.params[0] >= 0, "param value should be positive"
        
  def set_prior( self, prior ):
    if prior is not None:
      assert hasattr(prior,'rand'), "prior need 'rand'"
      assert hasattr(prior,'logdensity'), "prior need 'logdensity'"
      assert hasattr(prior,'logdensity_grad_free_x'), "prior need 'logdensity_grad_free_x'"
    self.prior = prior
    
  def get_range_of_params( self ):
    L =  0*np.ones( self.get_nbr_params() )
    R =  np.inf*np.ones( self.get_nbr_params() )
    stepsizes =  np.ones( self.get_nbr_params() )
    
    if self.prior is not None:
      return self.prior.get_range_of_params()
    else:
      return L,R,stepsizes
              
  def var( self, x = None ):
    if x is None:
      return self.params[0]
    
    self.check_inputs( x )
  
    [N,D] = x.shape
    return self.params[0]*np.eye(N)
      
  # assumes free parameters...
  def jacobians( self, K, X ):
    N = len(X)
    D = self.get_nbr_params()
    J = np.zeros( (N,N,D) )
    J[:,:,0] = self.params[0]*np.eye(len(X))
    return J