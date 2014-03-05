import numpy as np
import scipy as sp
import pylab as pp
from progapy.priors.composite_prior import CompositePrior
from progapy.priors.empty_prior import EmptyPrior
from progapy.priors.igamma_distribution import InverseGammaDistribution
from progapy.priors.gamma_distribution import GammaDistribution
from progapy.priors.normal_distribution import NormalDistribution

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
p1 = GammaDistribution( np.array([0.1,0.1]), np.array([0]) )
p2 = InverseGammaDistribution( np.array([0.1,0.1]), np.array([1]) )
kernel_prior = CompositePrior( [p1,p2] )
kernel = Kernel(kernel_params, kernel_prior)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# NOISE    ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #
noise_params = np.array([0.01])
noise_prior  = None
noise_prior = GammaDistribution( np.array([0.5,0.5]), np.array([0]) )
noise = Noise(noise_params, noise_prior)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# MEAN     ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #
mean_params = np.array( [np.random.randn()])
mean_prior  = NormalDistribution( np.array([0.0,0.1]), np.array([0]) )
mean = Mean(mean_params, mean_prior)
# --------------------------------------------------------------------------- #


N = 15
trainX, trainY = generate_data( N )
paramsDict = {"kernel":kernel, "noise":noise, "mean":mean}
gp = GP( paramsDict, trainX, trainY )

print gp.logposterior()

gp.check_grad( e = 1e-6 )
gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
pp.close('all')
pp.figure(1)
pp.clf()
view_this_gp( gp, x_range = [-1.5,1.5] )
pp.axis( [-1.25, 1.25, -3, 3])
# 
np.random.seed(2)
stepwidth = 0.01
nsamples = 36
thetas = gp.sample( method = "slice", params = {"nbrSteps":3,"N":16,"MODE":2})

pp.figure(2)
pp.clf()
for i in range(16):
  pp.subplot(4,4,i+1)
  gp.set_params(thetas[i])
  view_this_gp( gp, x_range = [-1.5,1.5] )
  pp.axis( [-1.25, 1.25, -2, 2])
pp.suptitle( "slice samples")
  
free_thetas = gp.sample( method = "hmc", params = {"L":50, "nsamples":16, "step_size":0.05})

pp.figure(3)
pp.clf()
for i in range(16):
  pp.subplot(4,4,i+1)
  gp.set_free_params(free_thetas[i])
  view_this_gp( gp, x_range = [-1.5,1.5] )
  pp.axis( [-1.25, 1.25, -2, 2])
pp.suptitle( "hmc samples")
pp.show()
