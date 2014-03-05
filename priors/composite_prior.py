import numpy as np
#from progapy.priors.prior_distribution import PriorDistribution

class CompositePrior( object ):
  
  def __init__( self, priors ):
    self.priors = priors
    self.n = 0
    for prior in self.priors:
      self.n += prior.n
      
  def set_params( self, params ):
    i = 0
    j = 0

    for prior in self.priors:
      # wrt the mean function parameters
      nbr = prior.n
      if nbr > 0:
        j += nbr
        prior.set_params( params[i:j])
        i = j
    
  # ========================================== #
  # required implementations by derived classes
  # ========================================== #
  def rand( self, N = 1 ):
    i = 0
    j = 0
    X = np.zeros( (self.n,N))
    for prior in self.priors:
      if prior.n > 0:
        j += prior.n
        X[i:j,:] = prior.rand( N )
        i = j
    return X
    
  def logdensity( self, x ):
    log_p = 0.0; i = 0; j = 0
    for prior in self.priors:
      if prior.n > 0:
        j += prior.n
        log_p += prior.logdensity( x[i:j] )
        i = j
    return log_p
    
  def logdensity_grad_free_x( self, free_x ):
    i = 0
    j = 0
    g = np.zeros(self.n)
    for prior in self.priors:
      if prior.n > 0:
        j += prior.n
        g[i:j] = prior.logdensity_grad_free_x( free_x[i:j] )
        i = j
    return g
    
  def get_range_of_params( self ):
    L = []
    R = []
    stepsizes = []
    
    for prior in self.priors:
      pL,pR,pstepsizes = prior.get_range_of_params()
      L.extend( pL )
      R.extend( pR )
      stepsizes.extend( pstepsizes )
    
    L = np.array(L)
    R = np.array(R)
    stepsizes = np.array(stepsizes)
    
    return L,R,stepsizes
    
  def get_range_of_free_params( self ):
    L = []
    R = []
    stepsizes = []
    
    for prior in self.priors:
      pL,pR,pstepsizes = prior.get_range_of_free_params()
      L.extend( pL )
      R.extend( pR )
      stepsizes.extend( pstepsizes )
    
    L = np.array(L)
    R = np.array(R)
    stepsizes = np.array(stepsizes)
    
    return L,R,stepsizes
    