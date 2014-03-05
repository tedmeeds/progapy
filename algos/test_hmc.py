import numpy as np
import scipy as sp
import pylab as pp
# from progapy.factories.json2gp import load_json, build_gp_from_json
# 
# from progapy.gps.basic_regression import BasicRegressionGaussianProcess as GP
# from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
# #from progapy.noises.fixed_noise_model import FixedNoiseModel as Noise
# from progapy.noises.standard_noise_model import StandardNoiseModel as Noise
# #from progapy.means.zero_mean_model import ZeroMeanModel as Mean
# from progapy.means.constant_mean_model import ConstantMeanModel as Mean
# 
# from progapy.viewers.view_1d import view as view_this_gp

from progapy.algos.hmc import *
#np.random.seed(0)
#from progapy.means.constant_mean_model import ConstantMeanModel as Mean

def gen_neglogprob( mu, cov ):
  D = len(mu)
  logdet = np.log( np.linalg.det(cov) )
  icov = np.linalg.inv( cov )
  constant = -0.5*np.log( 2*np.pi ) - 0.5*logdet
  def neglogprob( x, other_params = None ):
    d = x-mu
    return -(constant - 0.5*np.dot( np.dot( d.T, icov), d ))
  return neglogprob
  
def gen_neglogprob_grad_wrt_free_params( mu, cov ):
  D = len(mu)
  logdet = np.log( np.linalg.det(cov) )
  icov = np.linalg.inv( cov )
  def neglogprob_grad_wrt_free_params( fp, other_params = None ):
    return np.dot( icov, fp-mu)
  return neglogprob_grad_wrt_free_params
    
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
  D = 2
  mu = np.zeros(D)
  cov = np.array( [[1.0,0.98],[0.98,1.0]])
  x = np.zeros(D)    

  neglogprob = gen_neglogprob( mu, cov )
  neglogprob_grad_wrt_free_params = gen_neglogprob_grad_wrt_free_params( mu, cov )
  
  hmc_params={}
  hmc_params["nsamples"]                 = 100
  hmc_params["neglog_prob_wrt_free_params"] = neglogprob
  hmc_params["neglog_grad_wrt_free_params"] = neglogprob_grad_wrt_free_params
  hmc_params["L"]                        = 10
  hmc_params["step_size"]                = np.array([0.1,0.1])
  
  other_params = None
  
  X = hmc(x, hmc_params, other_params)
  
  pp.figure(1)
  pp.clf()
  pp.plot( X[:,0],X[:,1], 'o')
  pp.axis('equal')
  pp.show()