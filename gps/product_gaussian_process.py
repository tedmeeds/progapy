import numpy as np
import pdb
import scipy.linalg as spla

import numpy as np
import pylab as pp
import pdb


#from progapy.algos import sample
#from progapy.algos import optimize
  
class ProductGaussianProcess( object ):
  def __init__( self, gps, trainX = None, trainY = None ):
    self.gps = gps
    self.nbr_gps = len(gps)
    self.N = 0
    self.X = None
    self.Y = None
    
    # create array for indexing gps
    self.n_params       = np.zeros( self.nbr_gps, dtype = int  )
    self.start_param_ids = np.zeros( self.nbr_gps, dtype = int  )
    self.end_param_ids   = np.zeros( self.nbr_gps, dtype = int  )
    start_id = 0
    end_id   = 0
    self.nbr_params = 0
    for gp_id, gp in zip( range(self.nbr_gps), self.gps):
      n = gp.get_nbr_params()
      end_id += n
      self.start_param_ids[gp_id] = start_id
      self.end_param_ids[gp_id]   = end_id
      self.n_params[ gp_id ]     =  n
      start_id += n
      self.nbr_params += n
    
    if trainX is not None:
      assert trainY is not None, "Must provide trainY too"
      self.init_with_this_data( trainX, trainY )
      
  def init_with_this_data( self, trainX, trainY, force_precomputes = True ):
    [Nx,Dx] = trainX.shape
    [Ny,Dy] = trainY.shape
    
    assert Nx==Ny, "require same nbr of X and Y"
    assert Dy == self.nbr_gps, "one for each gp"
    
    self.N = Nx; self.D = Dx
    
    print "TODO: we are copying data, should we?"
    self.X = trainX.copy()
    self.Y = trainY.copy()
    
    for gp_id, gp in zip( range(self.nbr_gps), self.gps ):
      gp.init_with_this_data( trainX, trainY[:,gp_id].reshape( (Nx,1) ), force_precomputes=force_precomputes )
    
    #self.precomputes()
  
  def add_data( self, newX, newY, force_precomputes = True ): 
    if self.N == 0:
       return self.init_with_this_data( newX, newY )
       
    [Nx,Dx] = newX.shape
    [Ny,Dy] = newY.shape
    
    assert Nx==Ny, "require same nbr of X and Y"
    assert Dy == self.nbr_gps, "one for each gp"
    
    self.X = np.vstack( (self.X, newX ))
    self.Y = np.vstack( (self.Y, newY ))
    
    self.N = len(self.X)
    
    for gp_id, gp in zip( range(self.nbr_gps), self.gps ):
      gp.add_data( newX, newY[:,gp_id].reshape( (Nx,1) ), force_precomputes=force_precomputes )
        
  def precomputes( self ):
    for gp in self.gps():
      self.precomputes()
      
  def full_posterior( self, testX, use_noise = True ):
    MU  = []
    COV = []
    for gp in self.gps:
      mu, cov = gp.full_posterior( self, testX, use_noise )
      MU.append( mu )
      COV.append( cov )
    MU = np.array(MU)
    COV = np.array(COV)
    return MU, COV
  
  def full_posterior_mean_and_data( self, testX ):
    MU  = []
    COV = []
    DCOV = []
    for gp in self.gps:
      mu, mu_cov, data_cov = gp.full_posterior_mean_and_data( testX )
      MU.append( mu )
      COV.append( mu_cov )
      DCOV.append( data_cov )
    MU   = np.array(MU)
    COV  = np.array(COV)
    DCOV = np.array(DCOV)
    return MU, COV, DCOV
      
  def posterior( self, testX, use_noise = True ):
    MU  = []
    COV = []
    for gp in self.gps:
      mu, cov = gp.posterior( self, testX, use_noise )
      MU.append( mu )
      COV.append( np.diag(cov) )
    MU = np.array(MU)
    COV = np.array(COV)
    return MU, COV
    
  def optimize( self, method, params ):

    from progapy.algos import optimize
    for gp in self.gps:
      if method == "minimize":
        optimize.optimize_gp_with_minimize( gp, params )
      else:
        assert False, "optimize method = %s does not exist."%(method)
  
  def sample( self, method, params ):
    from progapy.algos import sample
    for gp in self.gps:
      if method == "slice":
        return sample.sample_gp_with_slice(gp, params)
      elif method == "hmc":
        return sample.sample_gp_with_hmc(gp, params)
      else:
        assert False, "sample method = %s does not exist."%(method)
      
  def check_grad( self, e, RETURNGRADS = False ):
    from progapy.algos import optimize
    for gp in self.gps:
      optimize.check_gp_gradients( gp, e, RETURNGRADS )
      
  def get_nbr_params(self):
    return self.nbr_params
      
  def get_params( self ):
    p = np.zeros( self.nbr_params )
    for a,b,gp in zip( self.start_param_ids, self.end_param_ids, self.gps ):
      p[a:b] = gp.get_params()
    return p
    
  def get_free_params( self ):
    fp = np.zeros( self.nbr_params )
    for a,b,gp in zip( self.start_param_ids, self.end_param_ids, self.gps ):
      fp[a:b] = gp.get_free_params()
    return fp
    
  def set_params( self, p ):
    for a,b,gp in zip( self.start_param_ids, self.end_param_ids, self.gps ):
      gp.set_params( p[a:b] )
  
  def set_free_params( self, fp ):
    for a,b,gp in zip( self.start_param_ids, self.end_param_ids, self.gps ):
      gp.set_free_params( fp[a:b] )
      
  def set_p_or_fp( self, p = None, fp = None ):
    if fp is not None:
      assert p is None, "only one"
      self.set_free_params( fp )
    elif p is not None:
      assert fp is None, "only one"
      self.set_params( p )
        
  def get_range_of_params( self ):
    LL=[]; RR=[]; S = []
    for gp in self.gps:
      L,R,stepsizes = gp.get_range_of_params()
      LL.extend(L)
      RR.extend(R)
      S.extend(stepsizes)
    return np.array(LL),np.array(RR),np.array(S)
    
  def neglogposterior( self, fp = None, p = None, X = None, Y = None ):
    return -self.logposterior( p=p, fp=fp, X = X, Y = Y )
    
  def logposterior( self, fp = None, p = None, X = None, Y = None ):
    self.set_p_or_fp( p=p, fp=fp )
      
    return self.loglikelihood( X = X, Y = Y ) + self.logprior()
      
  def loglikelihood( self, p = None, fp = None, X = None, Y = None ):
    self.set_p_or_fp( p, fp )
    
    log_p = self.data_loglikelihood(X,Y)
    
    assert log_p.__class__ != np.array, "make sure every log prob return float"
    return log_p

  def logprior( self, p = None, fp = None ):
    self.set_p_or_fp( p, fp )
    
    log_p  = 0
    for gp in self.gps:
      log_p += gp.logprior()
    
    assert log_p.__class__ != np.array, "make sure every log prob return float"
    return log_p
  
  def neglogposterior_grad_wrt_free_params( self, fp = None ):
    self.set_p_or_fp( fp = fp )
    return -self.logposterior_grad_wrt_free_params()
    
  def logposterior_grad_wrt_free_params( self, fp = None ):
    self.set_p_or_fp( fp = fp )
    return self.loglikelihood_grad_wrt_free_params() + self.logprior_grad_wrt_freeparams()
                
  def loglikelihood_grad_wrt_free_params( self, fp = None ):
    self.set_p_or_fp( fp = fp )
    
    g = np.zeros( (self.nbr_params(),1))
    for a,b,gp in zip( self.start_param_ids, self.end_param_ids, self.gps ):
      g[a:b,:] = gp.loglikelihood_grad_wrt_free_params()
    
    return g
  
  def logprior_grad_wrt_freeparams( self, fp = None ):
    if fp is not None:
      given = len(fp)
      nbr_params = self.get_nbr_params()
      assert given == nbr_params, "not correct nbr params given"
      self.set_free_params( fp )
    
    g = np.zeros( (self.nbr_params(),1))
    for a,b,gp in zip( self.start_param_ids, self.end_param_ids, self.gps ):
      g[a:b,:] = gp.logprior_grad_wrt_free_params()
  
    return g
      
  def data_loglikelihood( self, X = None, Y = None ):
    log_p = 0.0
    for gp_id, gp in zip( range(self.nbr_gps), self.gps ):
      if Y is not None:
        log_p  += gp.data_loglikelihood(X = X, Y = Y[:,gp_id])
      else:
        log_p  += gp.data_loglikelihood(X = X, Y = Y )
    return log_p