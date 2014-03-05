from slice import slice_sample
import numpy as np
from progapy.gp import generate_gp_logposterior_using_free_params as logposterior_generating_function_wrt_free_params
from progapy.gp import generate_gp_logposterior_using_params as logposterior_generating_function_wrt_params      
# params = {"L":0,"R":np.inf,"W":stepwidth,"N":nsamples,"MODE":2)}
# logdist,params,xinit,L,R,W,N,MODE
def sample_gp_with_slice( gp, params ):
  p = gp.get_params()
  #fp = gp.get_free_params()
  thetas = np.zeros( (params["N"],len(gp.get_params())))
  
  if params.has_key("ids"):
    ids = params["ids"]
  else:
    ids = range(len(gp.get_params()))
      
  cur_p = gp.get_params()
  #cur_fp = gp.get_free_params()
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