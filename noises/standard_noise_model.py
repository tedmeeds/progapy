import numpy as np
from progapy.noises import FixedNoiseModel

class StandardNoiseModel( FixedNoiseModel ):
    
  def get_nbr_params( self ):
    return len(self.params) 
   

  def logprior( self ):
    if self.prior is not None:
      return self.prior.logdensity()
    return 0
       
  def g_free_params( self, free_params, gp ):
    if self.priors is None:
      return self.params*gp.g_noise()
    else:
      return self.params*gp.g_noise() + self.params*self.priors.g()
    
  def g_params( self, params, gp ):
    if self.priors is None:
      return gp.g_noise()
    else:
      return gp.g_noise() + self.priors.g()