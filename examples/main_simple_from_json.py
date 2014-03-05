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
    
filename = "./examples/gp_1d.json"
json_gp = load_json( filename )
  

N = 5

np.random.seed(2)
trainX, trainY = generate_data( N )
gp = build_gp_from_json( json_gp )
gp.init_with_this_data( trainX, trainY )  

print gp.logposterior()
#gp.set_params( np.array([1,2.0,0.01]))

print gp.logposterior(X = trainX, Y = trainY)

gp.check_grad( e = 1e-6 )
gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
pp.figure(1)
pp.clf()
view_this_gp( gp, x_range = [-1.5,1.5] )
pp.axis( [-1.25, 1.25, -3, 3])

stepwidth = 0.2
nsamples = 49
thetas = gp.sample( method = "slice", params = {"N":nsamples,"MODE":2,"nbrSteps":3})

pp.figure(2)
pp.clf()
for i in range(nsamples):
  pp.subplot(np.sqrt(nsamples),np.sqrt(nsamples),i+1)
  gp.set_params(thetas[i])
  view_this_gp( gp, x_range = [-1.5,1.5] )
  pp.axis( [-1.25, 1.25, -2, 2])
pp.show()

pp.show()
