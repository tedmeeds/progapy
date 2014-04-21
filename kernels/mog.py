import numpy as np
import pylab as pp
import scipy.linalg as spla
from progapy.helpers import fast_distance, fast_grad_distance, logsumexp
from progapy.kernel import KernelFunction
import pdb
log2pi = np.log(2*np.pi)

def bagdata( nbrbags, X, Y = None ):
  N,D = X.shape
  
  p = []
  bagged = []
  for i in range(nbrbags):
    p.extend( np.arange(N) )
  p=np.array(p)[ np.random.permutation( N*nbrbags ) ].reshape( (N,nbrbags) )
  
  if Y is not None:
    baggedY = []
  for i in range(nbrbags):
    bagged.append( X[p[:,i],:].copy() )
    if Y is not None:
      baggedY.append( Y[p[:,i],:].copy() )
      
  if Y is not None:
    return bagged,baggedY
  else:
    return bagged
  
def loglike_mog( X, mu, cov, invcov, logdetcov ):
  N,D = X.shape

  if D > 1:
    lpdf = -0.5*D*log2pi - 0.5*logdetcov

    d = X-mu
    lpdf -= 0.5*np.sum( np.dot( d, invcov )*d, 1 )
  else:
    lpdf = -0.5*log2pi - 0.5*logdetcov

    d = X-mu
    lpdf -= 0.5*d*d/cov[0]
    lpdf = lpdf.reshape( (N,))
  return lpdf
    
def mog_logliks(X, PI, M, COV ):
  N,D = X.shape
  K = len(PI)
  R = np.zeros( ( N, K ) )
  logliks = np.zeros( ( N, K ) )
  for k in range( K ):
    logliks[:,k] = loglike_mog( X, M[k], COV[k] )

  ls = logsum( np.log(PI) + logliks,1 ).reshape( (len(X),1))
  return ls
  
def mog_assignment( X, PI, M, COV, INVCOV, LOGDETCOV ):
  N,D = X.shape
  K = len(PI)
  
  R = np.zeros( ( N, K ) )
  logliks = np.zeros( ( N, K ))
  for k in range( K ):
    logliks[:,k] = loglike_mog( X, M[k], COV[k], INVCOV[k], LOGDETCOV[k] )

  ls = logsumexp( np.log(PI) + logliks,1 ).reshape( (N,1))
  R = np.exp( np.log(PI) + logliks  - ls )
  return R, ls

def mog_update( X, R, K, priorPi = 0, factor = 0, mu0 = None, cov0 = None):    
  N,D = X.shape
  M   = np.zeros( (K,D) )
  COV = np.zeros( (K,D,D) )
  INVCOV = np.zeros( (K,D,D) )
  PI  = np.zeros(K)
  LOGDETCOV  = np.zeros(K)

  A = np.argmax(R,1)
  for k in range(K):
    Nk = R[:,k].sum()
    PI[k] = (priorPi + Nk) / N 
    M[k,:] =  np.sum( X*R[:,k].reshape((N,1)),0 ) / (priorPi + Nk)
    if mu0 is not None:
      M[k,:] = (1-factor)*M[k,:] + factor*mu0
    I = pp.find(A==k)
    d = (X - M[k,:])*pow(R[:,k].reshape((N,1)),0.5)
    #COV[k,:,:] = np.dot( d.T, d ) / len(I)

    COV[k,:,:] = np.dot( d.T, d) / (priorPi + Nk) + 1e-5*np.eye(D)
    if mu0 is not None:
      COV[k,:,:] = COV[k,:,:] + factor*cov0
    INVCOV[k,:,:] = np.linalg.inv( COV[k,:,:] )
    LOGDETCOV[k] = np.log( np.linalg.det( COV[k,:,:] ) )
    #A = np.argmax(R,1)
    #COV[k,:,:] = 0*np.dot( d.T, R[:,k].reshape( (len(d),1) )*d ) / Nk + np.eye(D)
  return M,COV,PI,INVCOV,LOGDETCOV
        
def kmeans_distance( X, M ):
  d = np.zeros( (len(X), len(M)))
  for k,m in zip( range(len(M)), M ):
    d[:,k] = np.sum( (X-m)**2, 1 )  
  
  if len(X)> 0:
    err = d.sum()/len(X)
  else:
    err = 0
  return d, err

def kmeans_assignment(X, M):
  d, err = kmeans_distance( X, M )
  A = np.argmin( d, 1 )
  R = np.zeros( (len(X),len(M) ) )
  for k in range( len(M) ):
    R[ pp.find(A==k),k] = 1
    
  return A, R, d, err

def kmeans_update( X, A, K ):
  N,D = X.shape
  M = np.zeros( (K,D) )
  for k in range(K):
    Ik=pp.find(A==k)
    if len(Ik) > 0:
      M[k,:] = X[ Ik,:].mean(0)
    else:
      M[k,:] = X.mean(0)
  return M
    
def kmeans( X, K ):
  N,D = X.shape
  M = X[ np.random.permutation(N)[:K],:]
  A, R, d, err = kmeans_assignment( X, M )
  #print "means error = %0.3f"%(err)
  if np.isnan(err):
    pdb.set_trace()
  bcontinue = True
  while bcontinue:
    M = kmeans_update( X, A, K )
    new_A, R, d, err = kmeans_assignment( X, M )
    dif = np.sum(A != new_A)
    if dif == 0:
      bcontinue = False
    A = new_A
    #print "means error = %0.3f  changes = %d"%(err,dif)
    if np.isnan(err):
      pdb.set_trace()
  return M
  
class MixtureOfGaussians(object):
  def __init__( self, K, X, priorPi = 1.0, factor = 0.1 ):
    self.X = X
    self.K = K
    self.em_eps = 1e-5
    self.priorPi = priorPi
    self.factor = factor
    self.mu0 = self.X.mean(0)
    self.cov0 = np.cov(self.X.T)

    self.log2pi = np.log(2*np.pi)
    self.logdets = [] # np.log(np.linalg.det(cov))

  def add_data( self, X, Y = None ):
    self.X = np.vstack( (self.X,X) )
    self.mu0 = self.X.mean(0)
    self.cov0 = np.cov(self.X.T)
    self.train( init = False, verbose = False )
    
  def view1d( self, xlim ):
    left = xlim[0]
    right = xlim[1]
    
    x = np.linspace( left, right, 1000 ).reshape( (1000,1))
    
    R, loglik = mog_assignment( x, self.PI, self.M, self.COV, self.INVCOV, self.LOGDETCOV )
    
    pp.plot( x,np.exp(loglik) )
      
  def train( self, init=True, verbose = False ):
    if init:
      if verbose:
        print "Running MOG ..."
        print "\t running kmeans... "
      self.M = kmeans( self.X, self.K )
      self.A, self.R, d, err = kmeans_assignment( self.X, self.M )
      if verbose:
        print "\t running mog first assignment "
      self.M, self.COV, self.PI, self.INVCOV, self.LOGDETCOV = mog_update( self.X, self.R, self.K, self.priorPi, self.factor, self.mu0, self.cov0 )
      
    #else:
    #  self.R, loglikall = mog_assignment( self.X, self.PI, self.M, self.COV, self.INVCOV, self.LOGDETCOV )
    #pdb.set_trace()

    old_loglik = -np.inf
    bcontinue = True
    while bcontinue:
      self.R, loglikall = mog_assignment( self.X, self.PI, self.M, self.COV, self.INVCOV, self.LOGDETCOV )
      self.loglik = loglikall.mean()
      self.M, self.COV, self.PI, self.INVCOV, self.LOGDETCOV = mog_update( self.X, self.R, self.K, self.priorPi, self.factor, self.mu0, self.cov0 )
  
      dif = self.loglik - old_loglik
      if dif < self.em_eps:
        bcontinue = False
      old_loglik = self.loglik
      if verbose:
        print "log_lik = %0.6f  changes = %0.6f"%(self.loglik,dif)
    return self.PI, self.M, self.COV
    #return PI, M, COV
    
  def class_conditional_posteriors( self, X ):
    R, loglik = mog_assignment( X, self.PI, self.M, self.COV, self.INVCOV, self.LOGDETCOV )
    return R

class BaggedMixtureOfGaussians(object):
  def __init__( self, K, X, Y, nBags, priorPi, factor ):
    self.trained_once = False
    self.X = X
    self.Y = Y
    self.K = K
    self.nBags = nBags
    self.em_eps = 1e-5
    self.priorPi = priorPi
    self.factor=factor

    self.N = 0
    self.D = 0
    if len(X) > 0:
      self.N,self.D = X.shape

  def view1d( self, xlim ):
    for mog in self.mogs:
      mog.view1d(xlim)

  def init_with_this_data( self, X, Y ):
    self.X = X
    self.Y = Y
    if len(X) > 0:
      self.N,self.D = X.shape
    
  def add_data( self, X, Y = None ):
    if self.trained_once == False:
      self.train()
      
    self.X = np.vstack( (self.X,X) )
    if Y is not None:
      self.Y = np.vstack( (self.Y,Y) )
    newbaggedX, newbaggedY = bagdata( self.nBags, self.X, self.Y )  
   
    for mog_id in range( self.nBags ):
      # add offset for indices
      self.baggedX[mog_id] = np.vstack( (self.baggedX[mog_id], newbaggedX[mog_id]) )
      self.baggedY[mog_id] = np.vstack( (self.baggedY[mog_id], newbaggedY[mog_id]) )
      self.mogs[mog_id].add_data( newbaggedX[mog_id] ) 
      
      Y_at_mog = self.baggedY[mog_id]*self.mogs[mog_id].class_conditional_posteriors(self.baggedX[mog_id] )
      self.amplitudes[mog_id] = np.var(Y_at_mog)
      
    self.N = len(self.X)
     
  def train( self, verbose = False ):
    self.mogs       = []
    self.amplitudes = []
    self.mean_y     = []
    self.baggedX, self.baggedY = bagdata( self.nBags, self.X, self.Y )
    
    for mog_id in range( self.nBags ):
      MOG = MixtureOfGaussians( self.K, self.baggedX[mog_id], self.priorPi, self.factor )
      MOG.train(verbose=verbose)
      self.mogs.append(MOG)
      var_at_mog = []
      mean_at_mog = []
      N_by_K_cond_dist = self.mogs[mog_id].class_conditional_posteriors(self.baggedX[mog_id])
      approxN = N_by_K_cond_dist.sum(0)
      for k in range(MOG.K):
        var_at_mog.append( np.std( self.baggedY[mog_id]*N_by_K_cond_dist[:,k]) )
        mean_at_mog.append( np.squeeze( np.dot(self.baggedY[mog_id].T,N_by_K_cond_dist[:,k]) ) )
      #pdb.set_trace()
      self.mean_y.extend(mean_at_mog/approxN)
      #pdb.set_trace()
      self.amplitudes.extend(var_at_mog)
    self.amplitudes = np.array(self.amplitudes)
    self.mean_y = np.array(self.mean_y)
    self.trained_once = True
  
  def update_train(self,verbose=False):  
    #return
    self.amplitudes = []
    for mog_id in range( self.nBags ):
      self.mogs[mog_id].train(init=False,verbose=verbose)
      var_at_mog = []
      N_by_K_cond_dist = self.mogs[mog_id].class_conditional_posteriors(self.baggedX[mog_id])
      for k in range(self.mogs[mog_id].K):
        var_at_mog.append( np.std( self.baggedY[mog_id]*N_by_K_cond_dist[:,k]) )
      #pdb.set_trace()
      self.amplitudes.extend(var_at_mog)
    self.amplitudes = np.array(self.amplitudes)
      
  def class_conditional_posteriors( self, X ):
    N,D = X.shape
    R = np.zeros( (N,self.K*self.nBags) )
    idx = 0
    for mog in self.mogs:
      mogR, mogloglik = mog_assignment( X, mog.PI, mog.M, mog.COV, mog.INVCOV, mog.LOGDETCOV )
      R[:,idx:idx+self.K] = mogR
      idx+=self.K
    
    return R
        
class MixtureOfGaussiansFunction( KernelFunction ):
  
  def set_params(self,params=None):
    self.params = params
    
  def train(self):
    print "MOG train"
    self.mog.train()
  
  def update_train(self):
    print "MOG update_train"
    self.mog.update_train()
    
  def init_with_this_data( self, X, Y ):
    self.mog.init_with_this_data(X,Y)
      
  def add_data( self,  X, Y = None ):
    self.mog.add_data( X, Y )
    if np.random.rand()<0.1:
      print "MOG update_train"
      self.mog.train()
      #self.mog.update_train()
    
  def set_mog( self, mog ):
    self.mog = mog
    
  # for now, mog is independent from GP
  def get_nbr_params( self ):
    return 0
    
  def shrink_length_scales(self, factor ):
    pass
        
  def compute_symmetric( self, params, X, with_self ):
    PHI = self.mog.class_conditional_posteriors( X )
    
    #NRM1 = np.sqrt( np.sum(PHI * PHI, 1) )
    return np.dot( PHI, PHI.T ) #/np.dot( NRM1, NRM1.T )

  def compute_asymmetric( self, params, X1, X2 ):
    PHI1 = self.mog.class_conditional_posteriors( X1 )
    PHI2 = self.mog.class_conditional_posteriors( X2 )
    
    #NRM1 = np.sqrt( np.sum(PHI1 * PHI1, 1) ).reshape( (len(X1),1))
    #NRM2 = np.sqrt( np.sum(PHI2 * PHI2, 1) ).reshape( (len(X2),1))
    return np.dot( PHI1, PHI2.T ) #/np.dot(NRM1, NRM2.T)
    
  # assumes free parameters...
  def jacobians( self, K, X ):
    pass
    
  def get_range_of_params( self ):
    pass
    
class WeightedMixtureOfGaussiansFunction( MixtureOfGaussiansFunction ):
  
        
  def compute_symmetric( self, params, X, with_self ):
    PHI = self.mog.amplitudes*self.mog.class_conditional_posteriors( X )
    
    #NRM1 = np.sqrt( np.sum(PHI * PHI, 1) )
    return np.dot( PHI, PHI.T ) #/np.dot( NRM1, NRM1.T )

  def compute_asymmetric( self, params, X1, X2 ):
    PHI1 = self.mog.amplitudes*self.mog.class_conditional_posteriors( X1 )
    PHI2 = self.mog.amplitudes*self.mog.class_conditional_posteriors( X2 )
    
    #pdb.set_trace()
    return np.dot( PHI1, PHI2.T ) #/np.dot(NRM1, NRM2.T)
    
    