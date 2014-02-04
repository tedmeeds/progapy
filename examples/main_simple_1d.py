import numpy as np
import scipy as sp
import pylab as pp

from progapy.gps.basic_regression import BasicRegressionGaussianProcess as GP
from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
#from progapy.noises.fixed_noise_model import FixedNoiseModel as Noise
from progapy.noises.standard_noise_model import StandardNoiseModel as Noise
#from progapy.means.zero_mean_model import ZeroMeanModel as Mean
from progapy.means.constant_mean_model import ConstantMeanModel as Mean

from progapy.viewers.view_1d import view as view_this_gp
#np.random.seed(0)
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
kernel_params = np.array([1.50, 0.25])
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
mean_params = np.array( [np.random.randn()])
mean_prior  = None
mean = Mean(mean_params, mean_prior)
# --------------------------------------------------------------------------- #


N = 3
trainX, trainY = generate_data( N )
paramsDict = {"kernel":kernel, "noise":noise, "mean":mean}
gp = GP( paramsDict, trainX, trainY )

print gp.marginal_loglikelihood()
#gp.set_params( np.array([1,2.0,0.01]))

print gp.marginal_loglikelihood(trainX, trainY)

gp.check_grad( e = 1e-6 )
gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
pp.figure(1)
pp.clf()
view_this_gp( gp, x_range = [-1.5,1.5] )
pp.axis( [-1.25, 1.25, -3, 3])

np.random.seed(2)
stepwidth = 0.01
nsamples = 36
thetas = gp.sample( method = "slice", params = {"L":0.0,"R":np.inf,"W":stepwidth,"N":nsamples,"MODE":2})

pp.figure(2)
pp.clf()
for i in range(36):
  pp.subplot(6,6,i+1)
  gp.set_params(thetas[i])
  view_this_gp( gp, x_range = [-1.5,1.5] )
  pp.axis( [-1.25, 1.25, -2, 2])
pp.show()
