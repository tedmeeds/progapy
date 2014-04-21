import numpy as np
import pylab as pp
import scipy.linalg as spla
from progapy.helpers import fast_distance, fast_grad_distance, logsumexp
#from progapy.kernel import KernelFunction
from progapy.kernels.mog import *
from progapy.mean import MeanModel
import pdb
log2pi = np.log(2*np.pi)


        
class MixtureOfGaussiansMeanModel( MeanModel ):
  
  def set_params(self,params=None):
    self.params = params
    
  def train(self):
    print "mean MOG train"
    self.mog.train()
  
  def update_train(self):
    print "mean MOG update_train"
    self.mog.update_train()
    
  def init_with_this_data( self, X, Y ):
    self.mog.init_with_this_data(X,Y)
      
  def add_data( self,  X, Y = None ):
    self.mog.add_data( X, Y )
    if np.random.rand()<0.1:
      print "mean MOG update_train"
      self.mog.train()
      #self.mog.update_train()
    
  def set_mog( self, mog ):
    self.mog = mog
    
  # for now, mog is independent from GP
  def get_nbr_params( self ):
    return 0
  def mu( self, x ):
    #pdb.set_trace()
    N,D = self.check_inputs( x )
    PHI = self.mog.class_conditional_posteriors( x )
    
    mu = np.dot( PHI, self.mog.mean_y ) / PHI.sum(1)
    #pdb.set_trace()
    #NRM1 = np.sqrt( np.sum(PHI * PHI, 1) )
    #print "mushape", mu.shape
    return mu.reshape((N,1))
    return mu #np.dot( PHI, PHI.T ) #/np.dot( NRM1, NRM1.T )
