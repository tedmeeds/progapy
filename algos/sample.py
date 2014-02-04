from slice import slice_sample
import numpy as np

def gp_grad_params_objective( w, gp ):
  return -gp.grad_free_p_objective(w)
  
def gp_grad_p( w, gp ):
  return -gp.grad_free_p(w ).squeeze()
  
def gp_func_x( x, gp ):
  x = x.reshape( (1,gp.dim))
  mu, cov = gp.full_posterior( x, False )
  return -mu
  
def gp_grad_x( x, gp ):
  x = x.reshape( (1,gp.dim))
  mu, cov = gp.full_posterior( x, True )
  
  grad_mu = gp.mu.grad_x( x, gp )
  
  alpha = np.dot( gp.inv_K_x_x, gp.ytrain - gp.mu.eval(gp.Xtrain) )
  
  kernel_grad_x = gp.kernel.grad_x_data( x, gp )
  
  return -(grad_mu + np.dot( kernel_grad_x.T, alpha )).squeeze()

def gen_logdist( gp ):
  def logdist( p, idx ):
    curp = gp.get_params()
    pp = curp.copy()
    pp[idx] = p
    gp.set_params( pp )
    return float(gp.marginal_loglikelihood())
  return logdist
      
# params = {"L":0,"R":np.inf,"W":stepwidth,"N":nsamples,"MODE":2)}
# logdist,params,xinit,L,R,W,N,MODE
def sample_gp_with_slice( gp, params ):
  p = gp.get_params()
  thetas = np.zeros( (params["N"],len(gp.get_free_params())))
  
  cur_p = gp.get_params()
  for n in range(params["N"]):
    for i in range( len(gp.get_free_params()) ):
      
      X, logprob = slice_sample( gen_logdist(gp), \
                               i, \
                               cur_p[i], \
                               params["L"], params["R"],params["W"],2,params["MODE"] )
      cur_p[i] = X[-1]
    thetas[n,:] = cur_p
    
  gp.set_params(p)
  return thetas