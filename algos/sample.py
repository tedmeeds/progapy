from slice import slice_sample
from hmc import hmc

import numpy as np
#from progapy.gp import generate_gp_logposterior_using_free_params as logposterior_generating_function_wrt_free_params
#from progapy.gp import generate_gp_logposterior_using_params as logposterior_generating_function_wrt_params  
#from progapy.gp import gp_neglogposterior_using_free_params    
#from progapy.gp import gp_neglogposterior_grad_wrt_free_params 


import progapy.gp

def sample_gp_with_slice( gp, params ):
  logposterior_generating_function_wrt_params = progapy.gp.generate_gp_logposterior_using_params
  
  p = gp.get_params()
  thetas = np.zeros( (params["N"],len(gp.get_params())))
  
  if params.has_key("ids"):
    ids = params["ids"]
  else:
    ids = range(len(gp.get_params()))
      
  cur_p = gp.get_params()
  L,R,stepsizes     = gp.get_range_of_params()

  for n in range(params["N"]):
    for i in ids:
      
      X, logprob = slice_sample( logposterior_generating_function_wrt_params(gp, idx=i), \
                               i, \
                               cur_p[i], \
                               L[i], \
                               R[i], \
                               stepsizes[i], \
                               params["nbrSteps"], \
                               params["MODE"] )
      cur_p[i] = X[-1]
    thetas[n,:] = cur_p
    
  gp.set_params(p)
  return thetas
  
def sample_gp_with_hmc( gp, params ):
  gp_neglogposterior_using_free_params = progapy.gp.gp_neglogposterior_using_free_params
  gp_neglogposterior_grad_wrt_free_params = progapy.gp.gp_neglogposterior_grad_wrt_free_params
  
  hmc_params={}
  hmc_params["nsamples"]                    = params["nsamples"]
  hmc_params["neglog_prob_wrt_free_params"] = gp_neglogposterior_using_free_params
  hmc_params["neglog_grad_wrt_free_params"] = gp_neglogposterior_grad_wrt_free_params
  hmc_params["L"]                           = params["L"]
  hmc_params["step_size"]                   = params["step_size"]
  
  other_params = gp
  
  fp = gp.get_free_params()
  FP = hmc( fp, hmc_params, other_params )
  
  gp.set_free_params(fp)
  return FP