import json
import numpy as np
from progapy.helpers import json_extract_from_list
from progapy.gps.basic_regression        import BasicRegressionGaussianProcess
from progapy.kernels.squared_exponential import SquaredExponentialFunction
from progapy.kernels.matern32            import Matern32Function
from progapy.kernels.matern52            import Matern52Function
from progapy.noises.fixed_noise_model    import FixedNoiseModel
from progapy.noises.standard_noise_model import StandardNoiseModel
from progapy.means.zero_mean_model       import ZeroMeanModel
from progapy.means.constant_mean_model   import ConstantMeanModel

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
    params["kernel"] = build_kernel( json_gp["kernel"] )
  else:
    params["kernel"] = build_default_kernel( dx, dy )
    
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
    params["mean"] = build_mean( json_gp["mean"] )
  else:
    params["mean"] = build_default_mean()
    
  # ----------------------------------------#
  # GP
  # ----------------------------------------#
  if json_gp.has_key("gp"):
    gp = build_gp( json_gp["gp"], params, X, Y  )
  else:
    gp = build_default_gp( params, X, Y )
    
  return gp
  
# --------------------------------------- #
# DEFAULTS FACTORIES
# --------------------------------------- #
def build_default_gp( params, X, Y ):
  return BasicRegressionGaussianProcess( params, X, Y )
  
def build_default_kernel( dx, dy ):
  params     = np.ones( dx+1 )
  params[0]  = DEFAULT_AMPLITUDE
  params[1:] = DEFAULT_LENGTH_SCALE
  
  return Matern32( params )
  
def build_default_noise( dy ):
  return FixedNoiseModel( DEFAULT_NOISE_VARIANCE )
  
def build_default_mean( dy ):
  return ZeroMeanModel()
  
  
# --------------------------------------- #
# SPECIFIC FACTORIES
# --------------------------------------- #
def build_kernel( json_kernel ):
  typeof = json_kernel["type"]
  params = json_kernel["params"]
  p=[]
  if typeof == "matern32":
    amp_value, amp_prior = json_extract_from_list( params, "name", "amp", ["value","prior"] )
    p.append(amp_value)
    ls_id = 1; ls_str = "ls_%d"%(ls_id)
    ls_value, ls_prior = json_extract_from_list( params, "name", ls_str, ["value","prior"] )
    p.append(ls_value)
    while ls_value is not None:
      ls_id += 1; ls_str = "ls_%d"%(ls_id)
      ls_value, ls_prior = json_extract_from_list( params, "name", ls_str, ["value","prior"] )
      if ls_value is not None:
        p.append(ls_value)
    k = Matern32Function( np.array(p) )
    
  elif typeof == "matern52":
    amp_value, amp_prior = json_extract_from_list( params, "name", "amp", ["value","prior"] )
    p.append(amp_value)
    ls_id = 1; ls_str = "ls_%d"%(ls_id)
    ls_value, ls_prior = json_extract_from_list( params, "name", ls_str, ["value","prior"] )
    p.append(ls_value)
    while ls_value is not None:
      ls_id += 1; ls_str = "ls_%d"%(ls_id)
      ls_value, ls_prior = json_extract_from_list( params, "name", ls_str, ["value","prior"] )
      if ls_value is not None:
        p.append(ls_value)
    k = Matern52Function( np.array(p) )
    
  elif typeof == "squared_exponential":
    amp_value, amp_prior = json_extract_from_list( params, "name", "amp", ["value","prior"] )
    p.append(amp_value)
    ls_id = 1; ls_str = "ls_%d"%(ls_id)
    ls_value, ls_prior = json_extract_from_list( params, "name", ls_str, ["value","prior"] )
    p.append(ls_value)
    while ls_value is not None:
      ls_id += 1; ls_str = "ls_%d"%(ls_id)
      ls_value, ls_prior = json_extract_from_list( params, "name", ls_str, ["value","prior"] )
      if ls_value is not None:
        p.append(ls_value)
    k = SquaredExponentialFunction( np.array(p) )
  
  else:
    raise NotImplementedError, "Have not implemented %s yet"%(typeof)
    
  return k
    
# --------------------------------------- #
# SPECIFIC FACTORIES : NOISE
# --------------------------------------- #
def build_noise( json_noise ):
  typeof = json_noise["type"]
  params = json_noise["params"]
  p=[]
  pr = []
  # "params" : [{"name" : "var", "value" : 0.1, "prior" : {"name" : "igamma", "params" : [1.0,1.0]}}]
  
  if typeof == "standard_noise_model":
    var_value, var_prior = json_extract_from_list( params, "name", "var", ["value","prior"] )
    p.append(var_value)
    pr.append( var_prior )
    
    noise = StandardNoiseModel( np.array( p ), var_prior )
    
  else:
    raise NotImplementedError, "Have not implemented %s yet"%(typeof)
  return noise
  
# --------------------------------------- #
# SPECIFIC FACTORIES : MEANS
# --------------------------------------- #
def build_mean( json_means ):
  typeof = json_means["type"]
  params = json_means["params"]
  p=[]
  pr = []
  # "params" : [{"name" : "var", "value" : 0.1, "prior" : {"name" : "igamma", "params" : [1.0,1.0]}}]
  #"name" : "mu", "value" : 1.0
  if typeof == "contant_mean_model":
    var_value = json_extract_from_list( params, "name", "mu", ["value"] )
    p.append(var_value)
    #pr.append( var_prior )
    
    means = ConstantMeanModel( np.array( p ) )
    
  else:
    raise NotImplementedError, "Have not implemented %s yet"%(typeof)
  return means
  
# --------------------------------------- #
# SPECIFIC FACTORIES : GPs
# --------------------------------------- #
def build_gp( typeof, params, X, Y ):
  if typeof == "basic_regression":
    gp = BasicRegressionGaussianProcess( params, X, Y )
  else:
    raise NotImplementedError, "Have not implemented %s yet"%(typeof)
  return gp
  
  
    