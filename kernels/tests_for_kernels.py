import numpy as np

#from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
#from progapy.kernels.matern52 import Matern52Function as Kernel
from progapy.kernels.matern32 import Matern32Function as Kernel
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

def jacobian_objective( fp, kernel, X ):
  kernel.set_free_params( fp )
  K = kernel.k( X )
  
  return K.sum()
  
def jacobian_gradient( fp, kernel, X ):
  g = np.zeros( len(fp) )
  
  J = kernel.jacobians( K, X )
  N,N,nbrP = J.shape
  for p in range(nbrP):
    g[p] = J[:,:,p].sum()
  
  return g

if __name__ == "__main__":
  N = 100
  e = 1e-3
  RETURNGRADS = False
  
  X,Y = generate_data( N )
  
  kernel = Kernel( np.array([ 0.75, 3.4 ]) )
  
  K = kernel.k( X )
  
  print checkgrad( jacobian_objective, jacobian_gradient, kernel.get_free_params(),e,RETURNGRADS,kernel, X)