#from progapy.gp import gp_neglogposterior_using_free_params as objective_function
#from progapy.gp import gp_neglogposterior_grad_wrt_free_params as grad_function

import progapy.gp

from minimize import minimize
from check_grad import checkgrad

def optimize_gp_with_minimize( gp, params ):
  objective_function = progapy.gp.gp_neglogposterior_using_free_params
  grad_function      = progapy.gp.gp_neglogposterior_grad_wrt_free_params
  
  best_p, v, t = minimize( gp.get_free_params(), \
                          objective_function, \
                          grad_function, \
                          [gp], \
                          maxnumlinesearch=params["maxnumlinesearch"] \
                         )
  print best_p
  gp.set_free_params( best_p )
  
def check_gp_gradients( gp, e, RETURNGRADS=False ):
  objective_function = progapy.gp.gp_neglogposterior_using_free_params
  grad_function      = progapy.gp.gp_neglogposterior_grad_wrt_free_params
  
  a = checkgrad( objective_function, \
                 grad_function, \
                 gp.get_free_params(), \
                 e, \
                 RETURNGRADS,\
                 gp\
               )
  print a
  return a