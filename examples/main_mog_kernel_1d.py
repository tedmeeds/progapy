import numpy as np
import scipy as sp
import pylab as pp
from progapy.priors.composite_prior import CompositePrior
from progapy.priors.empty_prior import EmptyPrior
from progapy.priors.igamma_distribution import InverseGammaDistribution
from progapy.priors.gamma_distribution import GammaDistribution
from progapy.priors.normal_distribution import NormalDistribution

from progapy.gps.basic_regression import BasicRegressionGaussianProcess as GP
from progapy.kernels.mog import MixtureOfGaussiansFunction as Kernel
from progapy.kernels.mog import WeightedMixtureOfGaussiansFunction as Kernel
from progapy.kernels.mog import MixtureOfGaussians as MOG
from progapy.kernels.mog import BaggedMixtureOfGaussians as BAGGEDMOG
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
  x = -1.5 + 2*np.random.rand(N)
  y = y_at_x(x) #np.sin(2*np.pi*(x+1) ) + 0.1*np.random.randn(N)
  
  x = x.reshape( (N,1) )
  y = y.reshape( (N,1) )
  
  return x,y
  
def y_at_x( x ):
  N = len(x)
  y = pow(5.0*x,3)*np.sin(2*np.pi*(x+1) ) # + 0.1*np.random.randn(N)
  
  y = y.reshape( (N,1) )
  return y
  
# --------------------------------------------------------------------------- #
    
# --------------------------------------------------------------------------- #
# KERNEL   ------------------------------------------------------------------ #
# --------------------------------------------------------------------------- #
#kernel_params = {"K":K,"nbags":nbags,"priorPi":priorPi}
kernel = Kernel( [], None )
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
mean_prior  = NormalDistribution( np.array([0.0,10.1]), np.array([0]) )
mean = Mean(mean_params, mean_prior)
# --------------------------------------------------------------------------- #

#np.random.seed(0)
N = 15
trainX, trainY = generate_data( N )
K = max(N,5)
mog = MOG( K, trainX )
nBags = 20
priorPi = 1.0/float(K)
factor=0.01
#mog = BAGGEDMOG( K, trainX, trainY, nBags, priorPi )
mog = BAGGEDMOG( K, [], [], nBags, priorPi, factor )
#mog.train()

kernel.set_mog(mog)

paramsDict = {"kernel":kernel, "noise":noise, "mean":mean}
gp = GP( paramsDict, trainX, trainY, [kernel], [kernel] )
#gp.subscribe_add_data( kernel )
#gp.subscribe_train( kernel )

gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )

x_plot = np.linspace( -1.5,1.5,100)
y_plot =  y_at_x(x_plot)
pp.close('all')
pp.figure(1)
pp.clf()
pp.subplot(2,1,1)
view_this_gp( gp, x_range = [-1.5,1.5] )
#pp.axis( [-1.5, 1.5, -3, 3])
pp.plot( x_plot, y_plot, 'r')
pp.xlim(-1.25,1.25)
#pp.ylim(-3,3)
pp.subplot(2,1,2)
mog.view1d((-1.5,1.5))
pp.xlim(-1.25,1.25)
# 
n=20
x_test =  0.3*np.random.randn(n) #-0.0*np.ones( n )
y_test = y_at_x(x_test)
x_test = x_test.reshape((n,1))
# 
gp.add_data( x_test, y_test )
# 
# 
pp.figure(2)
pp.clf()
pp.subplot(2,1,1)
view_this_gp( gp, x_range = [-1.5,1.5] )
pp.plot(x_test,y_test,'ro')
pp.plot( x_plot, y_plot, 'r')
pp.xlim(-1.25,1.25)
#pp.ylim(-3,3)
pp.subplot(2,1,2)
mog.view1d((-1.5,1.5))
pp.xlim(-1.25,1.25)

# pp.axis( [-1.25, 1.25, -3, 3])
# print gp.logposterior()
# 
# gp.check_grad( e = 1e-6 )
# gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
# pp.close('all')
# pp.figure(1)
# pp.clf()
# view_this_gp( gp, x_range = [-1.5,1.5] )
# pp.axis( [-1.25, 1.25, -3, 3])
# # 
# np.random.seed(2)
# stepwidth = 0.01
# nsamples = 36
# thetas = gp.sample( method = "slice", params = {"nbrSteps":3,"N":16,"MODE":2})
# 
# pp.figure(2)
# pp.clf()
# for i in range(16):
#   pp.subplot(4,4,i+1)
#   gp.set_params(thetas[i])
#   view_this_gp( gp, x_range = [-1.5,1.5] )
#   pp.axis( [-1.25, 1.25, -2, 2])
# pp.suptitle( "slice samples")
#   
# free_thetas = gp.sample( method = "hmc", params = {"L":50, "nsamples":16, "step_size":0.05})
# 
# pp.figure(3)
# pp.clf()
# for i in range(16):
#   pp.subplot(4,4,i+1)
#   gp.set_free_params(free_thetas[i])
#   view_this_gp( gp, x_range = [-1.5,1.5] )
#   pp.axis( [-1.25, 1.25, -2, 2])
# pp.suptitle( "hmc samples")
pp.show()
