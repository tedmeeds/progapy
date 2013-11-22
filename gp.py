

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
    
    self.X = trainX.copy()
    self.Y = trainY.copy()
    
    self.precomputes()
    
  def precomputes( self ):
    raise NotImplementedError
    
  def marginal_loglikelihood( self ):
    raise NotImplementedError
    
  def n_loglikelihood( self ):
    raise NotImplementedError
    
  def grad_n_loglike_wrt_free_params( self, free_params ):
    raise NotImplementedError