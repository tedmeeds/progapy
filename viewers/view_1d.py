import numpy as np
import scipy as sp
import pylab as pp
import pdb

def view( gp, x_range = None, N = 1000) : #, fignum = 1 ):
  
  # set range for x_test
  if x_range is None:
    X = gp.X
    x_range = [min(X), max(X)]
    
    if x_range[0] < 0:
      x_range[0] *= 1.25
    else:
      x_range[0] *= 0.75
    
    if x_range[1] < 0:
      x_range[1] *= 0.75
    else:
      x_range[1] *= 1.25
  
  print x_range
  
  testX = np.linspace( x_range[0], x_range[1], N ).reshape( (N,1) )
  f_test, fcov_test = gp.full_posterior( testX, use_noise = False )
  y_test, ycov_test = gp.full_posterior( testX, use_noise = True )
  
  f_test     = np.squeeze( f_test )
  testX      = np.squeeze( testX )
  f_sig_test = np.sqrt( np.diag(fcov_test) )
  y_sig_test = np.sqrt( np.diag(ycov_test) )
  
  if np.any(np.isnan(f_sig_test)):
    print f_sig_test
    pdb.set_trace()
  #pp.figure( fignum )
  #pp.clf()
  
  # full model + noise stddevs
  pp.fill_between( testX, f_test - 2*y_sig_test, f_test + 2*y_sig_test, color = 'r', alpha=0.5 )
  
  # model stds
  pp.fill_between( testX, f_test - 2*f_sig_test, f_test + 2*f_sig_test, color = 'b', alpha=0.75 )
  
  pp.plot( testX, f_test + 2*y_sig_test, "r-", lw=0.75)
  pp.plot( testX, f_test - 2*y_sig_test, "r-", lw=0.75)
  pp.plot( testX, f_test + 2*f_sig_test, "b-", lw=0.75)
  pp.plot( testX, f_test - 2*f_sig_test, "b-", lw=0.75)
  pp.plot( testX, f_test, "k-", lw=2)
  
  pp.plot( gp.X, gp.Y, 'yo' )