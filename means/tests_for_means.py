import numpy as np

#from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
#from progapy.kernels.matern52 import Matern52Function as Kernel
#from progapy.noises.standard_noise_model import StandardNoiseModel as Noise
from progapy.algos.check_grad import checkgrad
from progapy.means.constant_mean_model import ConstantMeanModel as Mean

# --------------------------------------------------------------------------- #
# SINUSOIDAL DATA   --------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def generate_data( N ):
  x = -1 + 2*np.random.rand(N)
  y = np.sin(2*np.pi*(x+1) ) + 0.1*np.random.randn(N)
  
  x = x.reshape( (N,1) )
  y = y.reshape( (N,1) )
  
  return x,y
# --------------------------------------------------------------------------- #

def grad_objective( fp, mean, X ):
  mean.set_free_params( fp )
  mu = mean.f(X)
  return mu.sum()
  
def grad_gradient( fp, mean, X ):
  g = np.zeros( len(fp) )
  grads = mean.grads( X )
  
  N,nbrP = grads.shape
  for p in range(nbrP):
    g[p] = grads[:,p].sum()
  
  return g

if __name__ == "__main__":
  N = 1000
  e = 1e-3
  RETURNGRADS = False
  
  X,Y = generate_data( N )
  
  # kernel = Kernel( np.array([ 0.75, 3.4 ]) )
  # noise  = Noise( np.array([ 0.45 ]) )
  mean   = Mean( np.array([-0.15]))
  
  # K = kernel.k( X )
  # C = K + noise.f(X)
  
  print checkgrad( grad_objective, grad_gradient, mean.get_free_params(),e,RETURNGRADS,mean, X)