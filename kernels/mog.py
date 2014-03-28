import numpy as np
import pylab as pp
import scipy.linalg as spla
from progapy.helpers import fast_distance, fast_grad_distance, logsumexp
from progapy.kernel import KernelFunction
import pdb
log2pi = np.log(2*np.pi)

def bagdata( X, nbrbags ):
  N,D = X.shape
  
  p = []
  bagged = []
  for i in range(nbrbags):
    p.extend( np.arange(N) )
  p=np.array(p)[ np.random.permutation( N*nbrbags ) ].reshape( (N,nbrbags) )
  
  for i in range(nbrbags):
    bagged.append( X[p[:,i],:].copy() )
  return bagged
  
def loglike_mog( X, mu, cov, invcov, logdetcov ):
  N,D = X.shape

  if D > 1:
    lpdf = -0.5*D*self.log2pi - 0.5*logdetcov

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

def mog_update( X, R, K, priorPi = 0):    
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
    I = pp.find(A==k)
    d = (X - M[k,:])*pow(R[:,k].reshape((N,1)),0.5)
    #COV[k,:,:] = np.dot( d.T, d ) / len(I)

    COV[k,:,:] = np.dot( d.T, d) / (priorPi + Nk) + 1e-5*np.eye(D)
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
  print "means error = %0.3f"%(err)
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
    print "means error = %0.3f  changes = %d"%(err,dif)
    if np.isnan(err):
      pdb.set_trace()
  return M
  
class MixtureOfGaussians(object):
  def __init__( self, K, X, priorPi = 1.0 ):
    self.X = X
    self.K = K
    self.em_eps = 1e-5
    self.priorPi = priorPi

    self.log2pi = np.log(2*np.pi)
    self.logdets = [] # np.log(np.linalg.det(cov))

  def add_data( self, X, Y = None ):
    self.X = np.vstack( (self.X,X) )
    self.train( init = False, verbose = True )
    
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
      self.M, self.COV, self.PI, self.INVCOV, self.LOGDETCOV = mog_update( self.X, self.R, self.K, self.priorPi )
      
    #else:
    #  self.R, loglikall = mog_assignment( self.X, self.PI, self.M, self.COV, self.INVCOV, self.LOGDETCOV )
    #pdb.set_trace()

    old_loglik = -np.inf
    bcontinue = True
    while bcontinue:
      self.R, loglikall = mog_assignment( self.X, self.PI, self.M, self.COV, self.INVCOV, self.LOGDETCOV )
      self.loglik = loglikall.mean()
      self.M, self.COV, self.PI, self.INVCOV, self.LOGDETCOV = mog_update( self.X, self.R, self.K, self.priorPi )
  
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
  def __init__( self, K, X, nBags, priorPi ):
    self.X = X
    self.K = K
    self.nBags = nBags
    self.em_eps = 1e-5
    self.N,self.D = X.shape
    self.priorPi = priorPi

  def view1d( self, xlim ):
    for mog in self.mogs:
      mog.view1d(xlim)

  def add_data( self, X, Y = None ):
    self.X = np.vstack( (self.X,X) )
    newbaggedX = bagdata( self.X, self.nBags )
   
    for mog_id in range( self.nBags ):
      # add offset for indices
      self.baggedX[mog_id] = np.vstack( (self.baggedX[mog_id], newbaggedX[mog_id]) )
      self.mogs[mog_id].add_data( newbaggedX[mog_id] ) 
   
    self.N = len(self.X)
     
  def train( self, verbose = False ):
    self.mogs = []
    self.baggedX = bagdata( self.X, self.nBags )
    
    for mog_id in range( self.nBags ):
      MOG = MixtureOfGaussians( self.K, self.baggedX[mog_id], self.priorPi )
      MOG.train()
      self.mogs.append(MOG)
    
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
  
  def set_mog( self, mog ):
    self.mog = mog
    
  # for now, mog is independent from GP
  def get_nbr_params( self ):
    return 0
    
  def shrink_length_scales(self, factor ):
    pass
        
  def compute_symmetric( self, params, X, with_self ):
    PHI = self.mog.class_conditional_posteriors( X )
    return np.dot( PHI, PHI.T )

  def compute_asymmetric( self, params, X1, X2 ):
    PHI1 = self.mog.class_conditional_posteriors( X1 )
    PHI2 = self.mog.class_conditional_posteriors( X2 )
    return np.dot( PHI1, PHI2.T )
    
  # assumes free parameters...
  def jacobians( self, K, X ):
    pass
    
  def get_range_of_params( self ):
    pass
    

    