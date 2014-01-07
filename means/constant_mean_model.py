import numpy as np
from progapy.mean import MeanModel

class ConstantMeanModel( MeanModel ):
    
  def check_params(self, params):
    assert len(params) == 1, "a single parameter; later when nbr output>0 ned to change this"
    
  def mu( self, x ):
    N,D = self.check_inputs( x )
    return self.params[0]*np.ones( (N,1) )
    
  def grads( self, X ):
    g = np.zeros( (len(X), self.get_nbr_params() ))
    g[:,0] = np.ones( (len(X), ) )
    return g