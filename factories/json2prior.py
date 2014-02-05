import numpy as np
from progapy.priors.igamma_distribution import InverseGammaDistribution

def build_prior( json_str ):
  if json_str is None:
    return None
    
  if json_str.has_key("name"):
    if json_str["name"] == "igamma":
      p = np.array( json_str["params"] )
      
      print "Building igamma prior with alpha = %f and beta = %f"%(p[0],p[1])
      
      prior = InverseGammaDistribution( p )
      
      return prior
    
    else:
      assert False, "Could not find 'name' in %s"%(str(json_str))
      
      