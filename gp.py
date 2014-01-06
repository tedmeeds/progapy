import numpy as np
from progapy.algos import optimize

class GaussianProcess( object ):
  def __init__( self, paramsDict, trainX = None, trainY = None ):
    self.kernel = paramsDict["kernel"] # 
    self.noise  = paramsDict["noise"]  # aka the model for observational noise
    self.mean   = paramsDict["mean"]   # aka the model for the prior mean function
    
    if trainX is not None:
      assert trainY is not None, "Must provide trainY too"
      self.init_with_this_data( trainX, trainY )
      
      
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
    
  def marginal_loglikelihood( self, X = None, Y = None ):
    raise NotImplementedError("This GP does not have marginal_loglikelihood!")
    
  def n_loglikelihood( self, X = None, Y = None ):
    raise NotImplementedError("This GP does not have n_loglikelihood!")
    
  def grad_n_loglike_wrt_free_params( self, free_params ):
    raise NotImplementedError("This GP does not have grad_n_loglike_wrt_free_params!")
    
  def full_posterior( self, testX, use_noise = True ):
    K_y_y = self.kernel.k( testX, with_self = True )
    
    if use_noise:
      K_y_y += self.noise.eval( testX )
      
    # if no training data, then use prior...
    if self.N == 0:
      mu = self.mean.eval(testX)
      return mu, K_y_y

    K_y_x = self.kernel.k( testX, self.X )
    
    Ly = np.linalg.solve( self.L, K_y_x.T )
    mu = self.mean.f(testX) + np.dot(Ly.T, self.mu_proj_y )
    
    cov = K_y_y - np.dot( Ly.T, Ly )
    return mu, cov
    
  def posterior( self, testX, use_noise = True ):
    mu,cov = self.full_posterior( testX, use_noise )
    return mu, np.diag(cov)
    
  def optimize( self, method, params ):
    if method == "minimize":
      optimize.optimize_gp_with_minimize( self, params )
    else:
      assert False, "optimize method = %s does not exist."%(method)
  
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
    
  def grad_free_p_objective( self, p ):
    #print "grad_p_objective received ", p, " size = ", p.shape
    self.set_free_params( p )
    
    log_p  = self.marginal_loglikelihood()
    log_p += self.mean.logprior()
    log_p += self.kernel.logprior()
    log_p += self.noise.logprior()
    
    try:
      return log_p[0,0]
    except:
      try:
        return log_p[0]
      except:
        return log_p
        
  def grad_free_p( self, p, typeof = "marginal" ):
    given = len(p)
    nbr_params = self.get_nbr_params()
    
    #print "given   = ",given
    #print "needed  = ",nbr_params
    assert given == nbr_params, "not correct nbr params given"
    
    self.set_free_params( p )
    
    g = np.zeros( (self.get_nbr_params(),1))
  
    i = 0
    j = 0
  
    # wrt the mean function parameters
    nbr_p_mu = self.mean.get_nbr_params()
    if nbr_p_mu > 0:
      j += nbr_p_mu
      g[i:j,:] = self.mean.g_free_params( self, typeof )
      i = j
  
    # wrt the kernel parameters
    j += self.kernel.get_nbr_params()  
    g[i:j,:] = self.kernel.g_free_params( self, typeof ).reshape( (j-i,1))
    i = j
  
    j += self.noise.get_nbr_params()
    if self.noise.get_nbr_params() > 0:
      g[i:j,:] = self.noise.g_free_params( self, typeof ).reshape( (j-i,1))
    
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
    
    
    
    
    
    
    
    
    
    