import numpy as np
import scipy as sp
import pylab as pp

from progapy.gps.basic_regression import BasicRegressionGaussianProcess as GP
from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
from progapy.noises.fixed_noise_model import FixedNoiseModel as Noise
from progapy.means.zero_mean_model import ZeroMeanModel as Mean

np.random.seed(0)
#from progapy.means.constant_mean_model import ConstantMeanModel as Mean

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
    
# --------------------------------------------------------------------------- #
# KERNEL   ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #
kernel_params = np.array([1.0, 0.25])
kernel_prior  = None
kernel = Kernel(kernel_params, kernel_prior)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# NOISE    ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #
noise_params = np.array([0.01])
noise_prior  = None
noise = Noise(noise_params, noise_prior)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# MEAN     ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #
mean_params = None
mean_prior  = None
mean = Mean(mean_params, mean_prior)
# --------------------------------------------------------------------------- #


N = 50
trainX, trainY = generate_data( N )
paramsDict = {"kernel":kernel, "noise":noise, "mean":mean}
gp = GP( paramsDict, trainX, trainY )

print gp.marginal_loglikelihood()
#gp.set_params( np.array([1,2.0,0.01]))

print gp.marginal_loglikelihood(trainX, trainY)

gp.check_grad( e = 1e-6 )
gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )

