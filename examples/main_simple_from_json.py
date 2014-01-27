import numpy as np
import scipy as sp
import pylab as pp

from progapy.factories.json2gp import load_json, build_gp_from_json

from progapy.gps.basic_regression import BasicRegressionGaussianProcess as GP
from progapy.kernels.squared_exponential import SquaredExponentialFunction as Kernel
#from progapy.noises.fixed_noise_model import FixedNoiseModel as Noise
from progapy.noises.standard_noise_model import StandardNoiseModel as Noise
#from progapy.means.zero_mean_model import ZeroMeanModel as Mean
from progapy.means.constant_mean_model import ConstantMeanModel as Mean

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
    
filename = "./examples/gp_1d.json"
json_gp = load_json( filename )
  

N = 50
trainX, trainY = generate_data( N )
gp = build_gp_from_json( json_gp )
gp.init_with_this_data( trainX, trainY )  

print gp.marginal_loglikelihood()
#gp.set_params( np.array([1,2.0,0.01]))

print gp.marginal_loglikelihood(trainX, trainY)

gp.check_grad( e = 1e-6 )
gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )

