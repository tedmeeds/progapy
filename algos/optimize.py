
from minimize import minimize
from check_grad import checkgrad

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
  
def optimize_gp_with_minimize( gp, params ):
  best_p, v, t = minimize( gp.get_free_params(), gp_grad_params_objective, gp_grad_p, [gp], maxnumlinesearch=params["maxnumlinesearch"] )
  print best_p
  gp.set_free_params( best_p )
  
def check_gp_gradients( gp, e, RETURNGRADS=False ):
  a = checkgrad( gp_grad_params_objective, gp_grad_p, gp.get_free_params(), e, RETURNGRADS,gp)
  print a
  return a