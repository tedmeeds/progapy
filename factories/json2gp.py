import json
import numpy as np
from progapy.helpers import json_extract_from_list
from progapy.gps.basic_regression        import BasicRegressionGaussianProcess
from progapy.kernels.squared_exponential import SquaredExponentialFunction
from progapy.kernels.matern32            import Matern32Function
from progapy.kernels.matern52            import Matern52Function
from progapy.kernels.mog                 import MixtureOfGaussiansFunction
from progapy.kernels.mog                 import WeightedMixtureOfGaussiansFunction
from progapy.noises.fixed_noise_model    import FixedNoiseModel
from progapy.noises.standard_noise_model import StandardNoiseModel
from progapy.means.zero_mean_model       import ZeroMeanModel
from progapy.means.constant_mean_model   import ConstantMeanModel
from progapy.means.mog                   import MixtureOfGaussiansMeanModel
#from progapy.kernels.mog import MixtureOfGaussiansFunction as Kernel
from progapy.kernels.mog import MixtureOfGaussians as MOG
from progapy.kernels.mog import BaggedMixtureOfGaussians as BAGGEDMOG

from progapy.factories.json2prior import build_prior, build_composite_prior

DEFAULT_NOISE_VARIANCE = 0.1
DEFAULT_AMPLITUDE      = 1.0
DEFAULT_LENGTH_SCALE   = 1.0

def load_json( filename ):
  fp = open( filename )
  json_gp = json.load( fp )
  fp.close()
  return json_gp
  
def build_gp_from_json( json_gp, deep = False ):
  # deep will copy the data too
  
  params = {}
  
  X = None; nx = 0; dx = 1 # by default, no examplars, but set-up a univariate input/output mapping
  Y = None; ny = 0; dy = 1
  # ----------------------------------------#
  # DATA
  # ----------------------------------------#
  if json_gp.has_key("X"):
    assert json_gp.has_key("Y"), "if it has X, must have Y"
    X = extract_data( json_gp["X"] )
    Y = extract_data( json_gp["Y"] )
    
    nx,dx = X.shape
    ny,dy = Y.shape
    
    assert nx == ny, "N for X and Y should be the same"
    
    N = nx
    
    # if we are not doing a deep copy, set these back to None
    if deep is False:
      X = None
      Y = None
      N = 0
      
  # ----------------------------------------#
  # KERNEL
  # ----------------------------------------#
  if json_gp.has_key("kernel"):
    params["kernel"], kernel_subscribe_add_data = build_kernel( json_gp["kernel"] )
  else:
    params["kernel"], kernel_subscribe_add_data = build_default_kernel( dx, dy )
    
  # ----------------------------------------#
  # NOISE
  # ----------------------------------------#
  if json_gp.has_key("noise"):
    params["noise"] = build_noise( json_gp["noise"] )
  else:
    params["noise"] = build_default_noise( dy )
    
  # ----------------------------------------#
  # MEAN
  # ----------------------------------------#
  if json_gp.has_key("mean"):
    params["mean"],mean_subscribe_add_data = build_mean( json_gp["mean"] )
  else:
    params["mean"],mean_subscribe_add_data = build_default_mean()
    
  # ----------------------------------------#
  # GP
  # ----------------------------------------#
  subscribed_to_listen_for_data = []
  subscribed_to_listen_for_train = []
  if kernel_subscribe_add_data:
    subscribed_to_listen_for_data.append( params["kernel"] )
    subscribed_to_listen_for_train.append( params["kernel"] )
  
  if mean_subscribe_add_data:
    subscribed_to_listen_for_data.append( params["mean"] )
    subscribed_to_listen_for_train.append( params["mean"] )  
  if json_gp.has_key("gp"):
    gp = build_gp( json_gp["gp"], params, X, Y, subscribed_to_listen_for_data, subscribed_to_listen_for_train  )
  else:
    gp = build_default_gp( params, X, Y, subscribed_to_listen_for_data, subscribed_to_listen_for_train )
    
    
  return gp
  
# --------------------------------------- #
# DEFAULTS FACTORIES
# --------------------------------------- #
def build_default_gp( params, X, Y, subscribed_to_listen_for_data=[], subscribed_to_listen_for_train=[] ):
  return BasicRegressionGaussianProcess( params, X, Y, subscribed_to_listen_for_data, subscribed_to_listen_for_train )
  
def build_default_kernel( dx, dy ):
  params     = np.ones( dx+1 )
  params[0]  = DEFAULT_AMPLITUDE
  params[1:] = DEFAULT_LENGTH_SCALE
  
  return Matern32( params ), False
  
def build_default_noise( dy = None ):
  return FixedNoiseModel( DEFAULT_NOISE_VARIANCE )
  
def build_default_mean( dy = None ):
  return ZeroMeanModel(), False
  
  
# --------------------------------------- #
# SPECIFIC FACTORIES
# --------------------------------------- #
def build_std_kernel_guts( params ):
  p = []
  priors = []
  idx = 0
  value, prior = json_extract_from_list( params, "name", "amp", ["value","prior"] )
  
  p.append(value)
  priors.append([prior, [idx]]); idx += 1
  
  ls_id = 1; ls_str = "ls_%d"%(ls_id)
  value, prior = json_extract_from_list( params, "name", ls_str, ["value","prior"] )
  p.append(value)
  priors.append([prior, [idx]]); idx += 1
  
  while value is not None:
    ls_id += 1; ls_str = "ls_%d"%(ls_id)
    value, prior = json_extract_from_list( params, "name", ls_str, ["value","prior"] )
    if value is not None:
      p.append(value)
      priors.append([prior, [idx]]); idx += 1
      
  p = np.array(p)
  prior = build_composite_prior( priors )
  assert len(p) == len(prior.priors), "we are enforcing a prior for each parameter"
  return p, prior
      
def build_kernel( json_kernel ):
  subscribe_add_data = False
  p=[]
  priors = []
  if json_kernel["type"] == "matern32":
    p,prior = build_std_kernel_guts( json_kernel["params"] )
    component = Matern32Function( p, prior )
    
  elif json_kernel["type"] == "matern52":
    p,prior = build_std_kernel_guts( json_kernel["params"] )
    component = Matern52Function( p, prior )
    
  elif json_kernel["type"] == "squared_exponential":
    p,prior = build_std_kernel_guts( json_kernel["params"] )
    component = SquaredExponentialFunction( p, prior )
    
  elif json_kernel["type"] == "mog":
    p,prior = build_std_kernel_guts( json_kernel["params"] )
    K, prior_def = json_extract_from_list( json_kernel["params"], "name", "K", ["value","prior"] )
    nbags, prior_def = json_extract_from_list( json_kernel["params"], "name", "nbags", ["value","prior"] )
    priorPi, prior_def = json_extract_from_list( json_kernel["params"], "name", "priorPi", ["value","prior"] )
    factor, prior_def = json_extract_from_list( json_kernel["params"], "name", "factor", ["value","prior"] )
    
    component = MixtureOfGaussiansFunction()
    mog = BAGGEDMOG( K, [], [], nbags, priorPi, factor )
    #mog.train()
    component.set_mog(mog)
    subscribe_add_data = True
    
  elif json_kernel["type"] == "wmog":
    p,prior = build_std_kernel_guts( json_kernel["params"] )
    K, prior_def = json_extract_from_list( json_kernel["params"], "name", "K", ["value","prior"] )
    nbags, prior_def = json_extract_from_list( json_kernel["params"], "name", "nbags", ["value","prior"] )
    priorPi, prior_def = json_extract_from_list( json_kernel["params"], "name", "priorPi", ["value","prior"] )
    factor, prior_def = json_extract_from_list( json_kernel["params"], "name", "factor", ["value","prior"] )
    
    component = WeightedMixtureOfGaussiansFunction()
    mog = BAGGEDMOG( K, [], [], nbags, priorPi, factor )
    #mog.train()
    component.set_mog(mog)
    subscribe_add_data = True
  
  else:
    raise NotImplementedError, "Have not implemented %s yet"%(typeof)
    
  return component, subscribe_add_data
    
# --------------------------------------- #
# SPECIFIC FACTORIES : NOISE
# --------------------------------------- #
def build_noise( json_noise ):
  idx = 0
  
  if json_noise["type"] == "standard_noise_model":
    value, prior_def = json_extract_from_list( json_noise["params"], "name", "var", ["value","prior"] )
    prior = build_prior( prior_def, np.array([idx]) )
    
    component = StandardNoiseModel( np.array( [value] ), prior )
  elif json_noise["type"] == "fixed_noise_model":
    value, prior_def = json_extract_from_list( json_noise["params"], "name", "var", ["value","prior"] )
    prior = build_prior( prior_def, np.array([idx]) )
    
    component = FixedNoiseModel( np.array( [value] ), prior )  
  else:
    raise NotImplementedError, "Have not implemented %s yet"%(typeof)
  return component
  
# --------------------------------------- #
# SPECIFIC FACTORIES : MEANS
# --------------------------------------- #
def build_mean( json_means ):
  idx = 0
  
  if json_means["type"] == "contant_mean_model":
    value, prior_def = json_extract_from_list( json_means["params"], "name", "mu", ["value","prior"] )
    prior = build_prior( prior_def, np.array([idx]) )
    
    component = ConstantMeanModel( np.array( [value] ), prior )
  elif json_means["type"] == "wmog":
    #value, prior_def = json_extract_from_list( json_means["params"], "name", "mu", ["value","prior"] )
    K, prior_def = json_extract_from_list( json_means["params"], "name", "K", ["value","prior"] )
    nbags, prior_def = json_extract_from_list( json_means["params"], "name", "nbags", ["value","prior"] )
    priorPi, prior_def = json_extract_from_list( json_means["params"], "name", "priorPi", ["value","prior"] )
    factor, prior_def = json_extract_from_list( json_means["params"], "name", "factor", ["value","prior"] )
    
    component = MixtureOfGaussiansMeanModel()
    mog = BAGGEDMOG( K, [], [], nbags, priorPi, factor )
    #mog.train()
    component.set_mog(mog)
    subscribe_add_data = True
    return component, True
    #prior = build_prior( prior_def, np.array([idx]) )
  
    #component = MixtureOfGaussiansMeanModel(  )
  else:
    raise NotImplementedError, "Have not implemented %s yet"%(typeof)
  return component, False
  
# --------------------------------------- #
# SPECIFIC FACTORIES : GPs
# --------------------------------------- #
def build_gp( typeof, params, X, Y, subscribed_to_listen_for_data=[], subscribed_to_listen_for_train=[] ):
  if typeof == "basic_regression":
    gp = BasicRegressionGaussianProcess( params, X, Y, subscribed_to_listen_for_data, subscribed_to_listen_for_train )
  else:
    raise NotImplementedError, "Have not implemented %s yet"%(typeof)
  return gp
  
  
    