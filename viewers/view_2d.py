import numpy as np
import scipy as sp
import pylab as pp
import pdb

def view( gp, x_range = None, y_range = None, N = 20 ) : #, fignum = 1 ):
  
  # set range for x_test
  if x_range is None:
    X = gp.X
    x_range = [min(X[:,0]), max(X[:,0])]
    y_range = [min(X[:,1]), max(X[:,1])]
    
    if x_range[0] < 0:
      x_range[0] *= 1.25
    else:
      x_range[0] *= 0.75
    
    if x_range[1] < 0:
      x_range[1] *= 0.75
    else:
      x_range[1] *= 1.25
    if y_range[0] < 0:
      y_range[0] *= 1.25
    else:
      y_range[0] *= 0.75
    
    if y_range[1] < 0:
      y_range[1] *= 0.75
    else:
      y_range[1] *= 1.25
  
  print x_range
  
  X = np.linspace( x_range[0], x_range[1], N )
  Y = np.linspace( y_range[0], y_range[1], N )
  
  testX = []
  for tx in X:
    for ty in Y:
      testX.append( [tx,ty])
  testX = np.array(testX)
  
  f_test, fcov_test = gp.full_posterior( testX, use_noise = False )
  y_test, ycov_test = gp.full_posterior( testX, use_noise = True )
  
  st = np.sqrt(np.diag(fcov_test))
  pp.subplot(1,2,1)
  pp.contourf( testX[:,0].reshape((N,N)), testX[:,1].reshape((N,N)), f_test.reshape((N,N)), 20  )
  pp.plot( gp.X[:,0], gp.X[:,1], 'ro' )
  pp.subplot(1,2,2)
  pp.contourf( testX[:,0].reshape((N,N)), testX[:,1].reshape((N,N)), st.reshape((N,N)), 20  )
  pp.plot( gp.X[:,0], gp.X[:,1], 'ro' )
  