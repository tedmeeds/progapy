import numpy as np

def hmc( current_free_params, hmc_params, other_params ):
  
  X = []
  x = current_free_params.copy()
  for n in range(hmc_params["nsamples"]):
    new_x = hmc_sample( x, hmc_params, other_params )
    X.append(new_x)
    x = new_x.copy()
  X = np.array(X)
  return X

def hmc_sample( current_free_params, hmc_params, other_params ):
  #gp_object  = params[0]
  neglogprob                  = hmc_params["neglog_prob_wrt_free_params"]
  neglog_grad_wrt_free_params = hmc_params["neglog_grad_wrt_free_params"]
  L                        = hmc_params["L"]
  step_size                = hmc_params["step_size"]
  
  q = current_free_params.copy()
  
  D = len(q)
  p = np.random.randn( D )
  current_p = p.copy()
  
  
  p = p - step_size*neglog_grad_wrt_free_params( q, other_params )/2.0
  
  for i in range(L):
    q = q + step_size*p
    if i + 1 < L:
      p = p - step_size*neglog_grad_wrt_free_params( q, other_params )
      
  p = p - step_size*neglog_grad_wrt_free_params(q, other_params)/2.0
  p = -p
  
  current_U = neglogprob(current_free_params, other_params )
  current_K = np.sum( current_p**2 )/2.0
  proposed_U = neglogprob( q, other_params )
  proposed_K = np.sum( p**2 )/2.0
  
  log_u = np.log( np.random.randn())
  log_acc = current_U - proposed_U + current_K - proposed_K
  #log_acc *= -1.0
  if log_acc > 0:
    x = q.copy()
  else:
    if np.random.randn() < np.exp( log_acc ):
      x = q.copy()
    else:
      x = current_free_params.copy()
  return x
    
  # if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K):
#     x = q
#   else:
#     x = current_q
  return q, current_U, proposed_U, current_K, proposed_K