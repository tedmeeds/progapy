import numpy as np
import pylab as pp

#from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
from progapy.priors.igamma_distribution import InverseGammaDistribution
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

def objective( fp, prior ):
  neg_loglik = -prior.logdensity( np.exp(fp) )
  return neg_loglik
  
def gradient( fp, prior ):
  neg_loglik_g = -prior.g_logdensity( np.exp(fp) )*np.exp(fp)
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
    #prior_params = np.array([ 3.0,1.0 ])
    leg.append( "a = %.1f b = %.1f"%(prior_params[0],prior_params[1]))
    prior = InverseGammaDistribution( prior_params )
  
  
    logp = prior.logdensity( p_range )
    
    pp.plot( p_range, np.exp(logp), c)
    
    print checkgrad( objective, gradient, params,e,RETURNGRADS,prior)
    
  pp.legend( leg)
  pp.ylabel("PDF")
  pp.xlabel("x")
  pp.title( "Inverse Gamma Distributions")
  pp.show()
  
  
  
    
if __name__ == "__main__":
  #N = 100
  test_igamma()