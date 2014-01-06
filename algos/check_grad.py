from numpy import *

def checkgrad(f,g,x,e,RETURNGRADS=False,*args):
    from pylab import norm
    """Check correctness of gradient function g at x by comparing to numerical
       approximation using perturbances of size e. Simple adaptation of 
       Carl Rasmussen's matlab-function checkgrad."""
    # print f
    # print g
    # print x
    # print e
    # print RETURNGRADS
    # print args
    
    dy = g(x,*args)
    if isscalar(x):
        dh = zeros(1,dtype=float)
        l = 1
    else:
        print "x in checkgrad:"
        print x 
        l = len(x)
        dh = zeros(l,dtype=float)
    for j in range(l):
        dx = zeros(l,dtype=float)
        dx[j] = e
        y2 = f(x+dx,*args)
        y1 = f(x-dx,*args)
        #print dx,y2,y1
        dh[j] = (y2 - y1)/(2*e)
        #print dh[j]
    print "analytic (using your gradient function): \n", dy
    print "approximation (using the objective function): \n", dh
    if RETURNGRADS: return dy,dh
    else: return norm(dh-dy)/norm(dh+dy)