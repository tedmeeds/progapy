import numpy as np

#from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
from progapy.kernels.matern52 import Matern52Function as Kernel
from progapy.noises.standard_noise_model import StandardNoiseModel as Noise
from progapy.algos.check_grad import checkgrad

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

def jacobian_objective( fp, noise, kernel, X ):
  noise.set_free_params( fp )
  K = kernel.k( X )
  C = K + noise.f(X)
  
  return C.sum()
  
def jacobian_gradient( fp, noise, kernel, X ):
  g = np.zeros( len(fp) )
  J = noise.jacobians( K, X )
  
  if len( J.shape ) == 3:
    N,N,nbrP = J.shape
    for p in range(nbrP):
      g[p] = J[:,:,p].sum()
  else:
    assert len(fp) == 1, "need more jacobians..."
    N,N = J.shape
    g[0] = J.sum()
  
  return g

if __name__ == "__main__":
  N = 100
  e = 1e-3
  RETURNGRADS = False
  
  X,Y = generate_data( N )
  
  kernel = Kernel( np.array([ 0.75, 3.4 ]) )
  noise  = Noise( np.array([ 0.45 ]) )
  
  K = kernel.k( X )
  C = K + noise.f(X)
  
  print checkgrad( jacobian_objective, jacobian_gradient, noise.get_free_params(),e,RETURNGRADS,noise,kernel, X)