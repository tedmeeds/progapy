from progapy.kernel import KernelFunction
from helpers import fast_distance

class SquaredExponentialFunction( KernelFunction ):
  
  def get_nbr_params( self ):
    return len(params)
    
  def compute( self, params, x1, x2 = None ):
    self.check_inputs( x1, x2 )
    
    [N1,D1] = x1.shape
    if x2 is None:
      return params[0]
    else:
      d = fast_distance( params[1:], x1, x2 )