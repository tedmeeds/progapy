import numpy as np
import pylab as pp

#from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
from progapy.priors.igamma_distribution import InverseGammaDistribution
from progapy.priors.gamma_distribution import GammaDistribution
from progapy.priors.normal_distribution import NormalDistribution
from progapy.algos.check_grad import checkgrad

# # --------------------------------------------------------------------------- #
# # SINUSOIDAL DATA   --------------------------------------------------------- #
# # --------------------------------------------------------------------------- #
# def generate_data( N ):
#   x = -1 + 2*np.random.rand(N)
#   y = np.sin(2*np.pi*(x+1) ) + 0.1*np.random.randn(N)
#   
#   x = x.reshape( (N,1) )
#   y = y.reshape( (N,1) )
#   
#   return x,y
# --------------------------------------------------------------------------- #

def objective( free_x, prior, exponentiate ):
  if exponentiate:
    x = np.exp(free_x)
  else:
    x = free_x
  neg_loglik = -prior.logdensity( x )
  return neg_loglik
  
def gradient( free_x, prior, exponentiate ):
  if exponentiate:
    x = np.exp(free_x)
  else:
    x = free_x
  neg_loglik_g = -prior.logdensity_grad_free_x( free_x )
  return neg_loglik_g

def test_igamma():
  e = 1e-3
  RETURNGRADS = False
  params = np.array( [2.3])
  fp=np.log(params)
  
  # plot wikepedia plots
  pp.figure(1)
  pp.clf()
  params2try = np.array([[1.0,1.0],[2.0,1.0],[3.0,1.0],[3.0,0.5]])
  colrs = ["r","g","b","m"]
  p_range = np.linspace( 1e-6,3.0,1000)
  leg = []
  for prior_params, c in zip( params2try, colrs ):
    leg.append( "a = %.1f b = %.1f"%(prior_params[0],prior_params[1]))
    prior = InverseGammaDistribution( prior_params )
    pp.plot( p_range, np.exp(prior.logdensity( p_range )), c,lw=2)
    print checkgrad( objective, gradient, np.log(params),e,RETURNGRADS,prior, True) 
  pp.legend( leg)
  pp.ylabel("PDF")
  pp.xlabel("x")
  pp.title( "Inverse Gamma Distributions")
  pp.show()

def test_gamma():
  e = 1e-3
  RETURNGRADS = False
  params = np.array( [2.3])
  fp=np.log(params)
  
  # plot wikepedia plots
  pp.figure(2)
  pp.clf()
  params2try = np.array([[1.0,0.5],[2.0,0.5],[3.0,0.5],[5.0,1.0],[9.0,2.0]])
  colrs = ["r","g","b","c","y"]
  p_range = np.linspace( 1e-6,20.0,1000)
  leg = []
  for prior_params, c in zip( params2try, colrs ):
    leg.append( "a = %.1f b = %.1f"%(prior_params[0],prior_params[1]))
    prior = GammaDistribution( prior_params )
    pp.plot( p_range, np.exp(prior.logdensity( p_range )), c,lw=2)
    print checkgrad( objective, gradient, np.log(params),e,RETURNGRADS,prior, True) 
  pp.legend( leg)
  pp.ylabel("PDF")
  pp.xlabel("x")
  pp.title( "Gamma Distributions")
  pp.show()  
  
def test_gaussian():
  e = 1e-3
  RETURNGRADS = False
  params = np.array( [2.3])
  fp=np.log(params)
  
  # plot wikepedia plots
  pp.figure(2)
  pp.clf()
  params2try = np.array([[1.0,0.5],[2.0,0.5],[0.0,0.5],[0.0,1.0],[0.0,2.0]])
  colrs = ["r","g","b","c","y"]
  p_range = np.linspace( -5,5.0,1000)
  leg = []
  for prior_params, c in zip( params2try, colrs ):
    leg.append( "a = %.1f b = %.1f"%(prior_params[0],prior_params[1]))
    prior = NormalDistribution( prior_params )
    pp.plot( p_range, np.exp(prior.logdensity( p_range )), c,lw=2)
    print checkgrad( objective, gradient, params,e,RETURNGRADS,prior, False) 
  pp.legend( leg)
  pp.ylabel("PDF")
  pp.xlabel("x")
  pp.title( "Gaussian Distributions")
  pp.show()  
    
if __name__ == "__main__":
  #N = 100
  test_igamma()
  #test_gamma()
  #test_gaussian()