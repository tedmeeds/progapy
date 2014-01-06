import numpy as np
import pdb
import scipy.linalg as spla
from progapy.gp import GaussianProcess

class BasicRegressionGaussianProcess( GaussianProcess ):
  def precomputes(self):
    self.centeredY    = self.Y - self.mean.mu( self.X )
    self.gram         = self.kernel.k( self.X )
    self.L            = np.linalg.cholesky( self.gram + self.noise.var( self.X ) ) 
    self.chol_solve_y = spla.cho_solve((self.L, True), self.centeredY ) 
    self.inv_K_x_x    = np.linalg.inv( self.gram + self.noise.var( self.X ) )
    self.K_x_x        = self.gram + self.noise.var( self.X ) 
    
  def marginal_loglikelihood( self, X = None, Y = None ):
    if X is None and Y is None:
      X = self.X
      Y = self.Y
      L = self.L
      chol_solve_y = self.chol_solve_y
      centeredY    = self.centeredY
    else:
      centeredY    = Y - self.mean.mu( X )
      gram         = self.kernel.k( X )
      L            = np.linalg.cholesky( gram + self.noise.var( X ) ) 
      chol_solve_y = spla.cho_solve( (L, True), centeredY ) 
      
    
    N = len(X)
    mll = -np.sum(np.log(np.diag(L)))-0.5*np.dot(centeredY.T, chol_solve_y)-0.5*N*np.log(2*np.pi)
    
    return np.squeeze(mll)
    
  def n_loglikelihood( self, X = None, Y = None ):
    return -self.marginal_loglikelihood( X, Y )
    
  def grad_n_loglike_wrt_free_params( self, free_params ):
    raise NotImplementedError("This GP does not have grad_n_loglike_wrt_free_params!")
    