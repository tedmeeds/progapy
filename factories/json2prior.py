import numpy as np
from progapy.priors.composite_prior import CompositePrior
from progapy.priors.empty_prior import EmptyPrior
from progapy.priors.igamma_distribution import InverseGammaDistribution
from progapy.priors.gamma_distribution import GammaDistribution
from progapy.priors.normal_distribution import NormalDistribution

def build_composite_prior( list_of_priors_and_ids ):
  priors = [build_prior( prior_def, ids ) for prior_def, ids in list_of_priors_and_ids]
  
  composite_prior = CompositePrior( priors )
  return composite_prior
  
def build_prior( json_str, ids ):
  if json_str is None:
    return EmptyPrior( None, ids )
    
  if json_str.has_key("name"):
    if json_str["name"] == "igamma":
      p = np.array( json_str["params"] )
      
      print "Building igamma prior with alpha = %f and beta = %f"%(p[0],p[1])
      
      prior = InverseGammaDistribution( p, ids )
      
      return prior

    elif json_str["name"] == "gamma":
      p = np.array( json_str["params"] )
      
      print "Building gamma prior with alpha = %f and beta = %f"%(p[0],p[1])
      
      prior = GammaDistribution( p, ids )
      
      return prior
      
    elif json_str["name"] == "gaussian" or json_str["name"] == "normal":
      p = np.array( json_str["params"] )
      
      print "Building Gaussian prior with mu = %f and var = %f"%(p[0],p[1])
      
      prior = NormalDistribution( p, ids )
      
      return prior
    
    else:
      assert False, "Could not find 'name' in %s"%(str(json_str))
      
      