import numpy as np
import pylab as pp
import pdb

# functions that take params first and additional params next (eg gp)
def gp_neglogposterior_using_free_params( fp, gp ):
  return gp.neglogposterior( fp = fp )
  
def gp_neglogposterior_grad_wrt_free_params( fp, gp ):
  return gp.neglogposterior_grad_wrt_free_params( fp = fp ).squeeze()

def generate_gp_logposterior_using_free_params( gp, idx=None ):
  def gp_logposterior_using_free_params( fp_at_idx, params = None ):
    if idx is not None:
      fp = gp.get_free_params()
      fp[idx] = fp_at_idx
      gp.set_free_params( fp )
    return gp.logposterior()
  return gp_logposterior_using_free_params

def generate_gp_logposterior_using_params( gp, idx=None ):
  def gp_logposterior_using_params( p_at_idx, params = None ):
    if idx is not None:
      p = gp.get_params()
      p[idx] = p_at_idx
      gp.set_params( p )
    return gp.logposterior()
  return gp_logposterior_using_params
    
def gp_func_x( x, gp ):
  x = x.reshape( (1,gp.dim))
  mu, cov = gp.full_posterior( x, False )
  return -mu
  
def gp_grad_x( x, gp ):
  x = x.reshape( (1,gp.dim))
  mu, cov = gp.full_posterior( x, True )
  
  grad_mu = gp.mu.grad_x( x, gp )
  
  alpha = np.dot( gp.inv_K_x_x, gp.ytrain - gp.mu.eval(gp.Xtrain) )
  
  kernel_grad_x = gp.kernel.grad_x_data( x, gp )
  
  return -(grad_mu + np.dot( kernel_grad_x.T, alpha )).squeeze()


from progapy.algos import optimize
from progapy.algos import sample
  
class GaussianProcess( object ):
  def __init__( self, paramsDict, trainX = None, trainY = None ):
    self.kernel = paramsDict["kernel"] # 
    self.noise  = paramsDict["noise"]  # aka the model for observational noise
    self.mean   = paramsDict["mean"]   # aka the model for the prior mean function
    
    if trainX is not None:
      assert trainY is not None, "Must provide trainY too"
      self.init_with_this_data( trainX, trainY )
    self.typeof = "marginal"  
      
  def init_with_this_data( self, trainX, trainY ):
    [Nx,Dx] = trainX.shape
    [Ny,Dy] = trainY.shape
    
    assert Nx==Ny, "require same nbr of X and Y"
    assert Dy == 1, "for now, only univariate output"
    
    self.N = Nx; self.D = Dx
    
    print "TODO: we are copying data, should we?"
    self.X = trainX.copy()
    self.Y = trainY.copy()
    
    self.precomputes()
    
  def precomputes( self ):
    raise NotImplementedError("This GP does not have precomputes!")
  #   
  # def marginal_loglikelihood( self, X = None, Y = None ):
  #   raise NotImplementedError("This GP does not have marginal_loglikelihood!")
  #   
  # def n_loglikelihood( self, X = None, Y = None ):
  #   raise NotImplementedError("This GP does not have n_loglikelihood!")
   #  
  # def grad_n_loglike_wrt_free_params( self, free_params ):
  #   raise NotImplementedError("This GP does not have grad_n_loglike_wrt_free_params!")
  #     
  def full_posterior( self, testX, use_noise = True ):
    K_y_y = self.kernel.k( testX, testX ) #with_self = False )
    #self.mu_proj_x = np.linalg.solve( self.L, self.Y - self.mean.f(self.X) )
    #pdb.set_trace()
    if use_noise:
      K_y_y += self.noise.var( testX )
      
    # if no training data, then use prior...
    if self.N == 0:
      mu = self.mean.eval(testX)
      return mu, K_y_y

    K_y_x = self.kernel.k( testX, self.X )
    
    Linv_dot_Kyx = np.linalg.solve( self.L, K_y_x.T )
    mu = self.mean.f(testX) + np.dot(Linv_dot_Kyx.T, self.Linv_dot_y )
    # 
    # K_y_x = self.kernel.eval( Xtest, self.Xtrain, is_1d = self.is_1d, symmetric = False )
    # 
    # Ly = np.linalg.solve( self.L_x_x, K_y_x.T )
    # mu = self.mu.eval(Xtest) + np.dot(Ly.T, self.mu_proj_x )
    # 
    #cov = K_y_y - np.dot( np.dot( K_y_x, self.inv_K_x_x ), K_y_x.T )
    #cov = K_y_y - np.dot( Ly.T, Ly )
    
    print "******************************************"
    print "*********************"
    print "******************************************"
    print "*********************"
    print "assert the predictive cov is positive!!!"
    print "*********************"
    print "******************************************"
    print "*********************"
    print "******************************************"
    cov = K_y_y - np.dot( Linv_dot_Kyx.T, Linv_dot_Kyx )
    cov2 = K_y_y - np.dot( np.dot( K_y_x, np.linalg.inv(self.gram + self.noise.var( self.X ) ) ), K_y_x.T )
    vr = np.diag(cov)
    vr2 = np.diag(cov2)
    vr3 = np.diag(K_y_y) - np.sum(Linv_dot_Kyx**2,axis=0)
    if np.any(vr<0):
      pdb.set_trace()
    return mu, cov
    
  def posterior( self, testX, use_noise = True ):
    mu,cov = self.full_posterior( testX, use_noise )
    return mu, np.diag(cov)
    
  def optimize( self, method, params ):
    if method == "minimize":
      optimize.optimize_gp_with_minimize( self, params )
    else:
      assert False, "optimize method = %s does not exist."%(method)
  
  def sample( self, method, params ):
    if method == "slice":
      return sample.sample_gp_with_slice(self, params)
    elif method == "hmc":
      return sample.sample_gp_with_hmc(self, params)
    else:
      assert False, "sample method = %s does not exist."%(method)
      
  def check_grad( self, e, RETURNGRADS = False ):
    optimize.check_gp_gradients( self, e, RETURNGRADS )
      
  def get_nbr_params(self):
    n = 0
    n += self.mean.get_nbr_params()
    n += self.kernel.get_nbr_params()
    n += self.noise.get_nbr_params()
    return n
      
  def get_params( self ):
    p = np.zeros( (self.get_nbr_params(),1))
    i = 0
    j = 0

    # wrt the mean function parameters
    nbr_p_mu = self.mean.get_nbr_params()
    if nbr_p_mu > 0:
      j += nbr_p_mu
      p[i:j,:] = self.mean.get_params().reshape( (j-i,1) )
      i = j

    # wrt the kernel parameters
    j += self.kernel.get_nbr_params()  
    p[i:j,:] = self.kernel.get_params().reshape( (j-i,1) )
    i = j

    #j += self.noise_model.get_nbr_params().reshape( (j-i,1) )
    #p[i:j,:] = self.noise_model.get_params().reshape( (j-i,1) )
    nbr_p_noise = self.noise.get_nbr_params()
    if nbr_p_noise > 0:
      j += nbr_p_noise
      p[i:j,:] = self.noise.get_params().reshape( (j-i,1) )
      i = j
    return p.squeeze()
    
  def get_free_params( self ):
    p = np.zeros( (self.get_nbr_params(),1))
    i = 0
    j = 0

    # wrt the mean function parameters
    nbr_p_mu = self.mean.get_nbr_params()
    if nbr_p_mu > 0:
      j += nbr_p_mu
      p[i:j,:] = self.mean.get_free_params().reshape( (j-i,1) )
      i = j

    # wrt the kernel parameters
    j += self.kernel.get_nbr_params()  
    p[i:j,:] = self.kernel.get_free_params().reshape( (j-i,1) )
    i = j

    j += self.noise.get_nbr_params()
    if self.noise.get_nbr_params() > 0:
      p[i:j,:] = self.noise.get_free_params().reshape( (j-i,1) )

    return p.squeeze()
    
  def set_p_or_fp( self, p = None, fp = None ):
    if fp is not None:
      assert len(fp) == self.get_nbr_params(), "incorrect fp given"
      self.set_free_params( fp )
    elif p is not None:
      assert len(p) == self.get_nbr_params(), "incorrect p given"
      self.set_params( p )
        
  def get_range_of_params( self ):
    # L,R,stepsizes
    L = -np.inf*np.ones( self.get_nbr_params() )
    R =  np.inf*np.ones( self.get_nbr_params() )
    stepsizes =  np.ones( self.get_nbr_params() )
    
    #L = []; R = []; stepsizes = []
    i = 0
    j = 0

    # wrt the mean function parameters
    nbr_p_mu = self.mean.get_nbr_params()
    if nbr_p_mu > 0:
      j += nbr_p_mu
      l,r,s = self.mean.prior.get_range_of_params()
      L[i:j],R[i:j],stepsizes[i:j] = l,r,s
      #L.extend(l); R.extend(r); stepsizes.extend(s)
      i = j

    # wrt the kernel parameters
    j += self.kernel.get_nbr_params()  
    l,r,s =  self.kernel.prior.get_range_of_params()
    L[i:j],R[i:j],stepsizes[i:j] = l,r,s
    #L.extend(l); R.extend(r); stepsizes.extend(s)
    i = j

    j += self.noise.get_nbr_params()
    if self.noise.get_nbr_params() > 0:
      l,r,s = self.noise.prior.get_range_of_params()
      L[i:j],R[i:j],stepsizes[i:j] = l,r,s
      #L.extend(l); R.extend(r); stepsizes.extend(s)
    
    L = np.array(L)
    R = np.array(R)
    stepsizes = np.array(stepsizes)  
    return L,R,stepsizes
    
  def neglogposterior( self, fp = None, p = None, X = None, Y = None ):
    return -self.logposterior( p=p, fp=fp, X = X, Y = Y )
    
  def logposterior( self, fp = None, p = None, X = None, Y = None ):
    self.set_p_or_fp( p=p, fp=fp )
      
    return self.loglikelihood( X = X, Y = Y ) + self.logprior()
      
  def loglikelihood( self, p = None, fp = None, X = None, Y = None ):
    self.set_p_or_fp( p, fp )
    
    log_p  = self.data_loglikelihood(X = X, Y = Y)
    
    assert log_p.__class__ != np.array, "make sure every log prob return float"
    return log_p

  def logprior( self, p = None, fp = None ):
    self.set_p_or_fp( p, fp )
    
    log_p  = 0
    log_p += self.mean.logprior()
    log_p += self.kernel.logprior()
    log_p += self.noise.logprior()
    
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
    
    g = np.zeros( (self.get_nbr_params(),1))
  
    i = 0
    j = 0
  
    # wrt the mean function parameters
    nbr_p_mu = self.mean.get_nbr_params()
    if nbr_p_mu > 0:
      j += nbr_p_mu
      g[i:j,:] = self.mean.loglikelihood_grad_wrt_free_params( self )
      i = j
  
    # wrt the kernel parameters
    j += self.kernel.get_nbr_params()  
    g[i:j,:] = self.kernel.loglikelihood_grad_wrt_free_params( self ).reshape( (j-i,1))
    i = j
  
    j += self.noise.get_nbr_params()
    if self.noise.get_nbr_params() > 0:
      g[i:j,:] = self.noise.loglikelihood_grad_wrt_free_params( self ).reshape( (j-i,1))
    
    return g
  
  def logprior_grad_wrt_freeparams( self, fp = None ):
    if fp is not None:
      given = len(fp)
      nbr_params = self.get_nbr_params()
      assert given == nbr_params, "not correct nbr params given"
      self.set_free_params( fp )
    
    g = np.zeros( (self.get_nbr_params(),1))
  
    i = 0
    j = 0
  
    # wrt the mean function parameters
    nbr_p_mu = self.mean.get_nbr_params()
    if nbr_p_mu > 0:
      j += nbr_p_mu
      g[i:j,:] = self.mean.logprior_grad_wrt_free_params().reshape( (j-i,1))
      i = j
  
    # wrt the kernel parameters
    j += self.kernel.get_nbr_params()  
    g[i:j,:] = self.kernel.logprior_grad_wrt_free_params().reshape( (j-i,1))
    i = j
  
    j += self.noise.get_nbr_params()
    if self.noise.get_nbr_params() > 0:
      g[i:j,:] = self.noise.logprior_grad_wrt_free_params().reshape( (j-i,1))
    
    return g
      
  def set_free_params( self, p ):
    self.p = p.copy()
    
    # indices into parameters
    i = 0
    j = 0
  
    # wrt the mean function parameters
    nbr_p_mu = self.mean.get_nbr_params()
    if nbr_p_mu > 0:
      j += nbr_p_mu
      self.mean.set_free_params( p[i:j] )
      i = j
  
    # wrt the kernel parameters
    j += self.kernel.get_nbr_params()  
    self.kernel.set_free_params( p[i:j] )
    i = j
  
    j += self.noise.get_nbr_params()
    if self.noise.get_nbr_params() > 0:
      self.noise.set_free_params(p[i:j])
      
    self.precomputes()

  def set_params( self, p ):
    self.p = p.copy()

    # indices into parameters
    i = 0
    j = 0

    # wrt the mean function parameters
    nbr_p_mu = self.mean.get_nbr_params()
    if nbr_p_mu > 0:
      j += nbr_p_mu
      print "setting mean to: ", p[i:j] 
      self.mean.set_params( p[i:j] )
      i = j

    # wrt the kernel parameters
    j += self.kernel.get_nbr_params() 
    print "setting kernel to: ", p[i:j] 
    self.kernel.set_params( p[i:j] )
    i = j

    j += self.noise.get_nbr_params()

    if self.noise.get_nbr_params() > 0:
      print "setting noise to: ", p[i:j]
      self.noise.set_params(p[i:j])

    self.precomputes()
    
    
    
    
    
    
    
    
    
    