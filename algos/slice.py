from numpy import zeros, log, ceil, arange, isnan, nan
from numpy.random import rand, exponential
 

def slice_sample(logdist,params,xinit,L,R,W,N,MODE):
    """
    % Description
    % -------------------
    % Use slice sampling to generate samples from any distribution over
    % a one dimensional space where the initial interval [l,r] is known.
    % Takes a one parameter function computing the log of the distribution,
    % or any function computing the log of a function proportional to the
    % distribution. Additional parameters and constants can be passed in
    % using the params structure.
    %
    % Syntax: x=slice_sample(logdist,params,xinit,l,r,n)
    % --------------------
    % * logdist: a function handle for a function computing the log ofa distribution of
    %           one parameter.
    % * params:  a structure of paramaeters and constants needed by logdist
    % * xinit:   initial point to start sampling from
    % * L:       the lower bound of the sampling interval
    % * R:       the upper bound of the sampling interval
    % * N:       the number of samples to draw
    %* MODE:    0 - perform shrinkage on the given interval
    %           1 - perform stepping out then shrinkage
    %*          2 - perform doubling then shrinkage
    """
    
    # print "Running Slice Sampler with:"
    # print "    params = ", params
    # print "    xinit  = ", xinit
    # print "    L      = ", L
    # print "    R      = ", R
    # print "    W      = ", W
    # print "    N      = ", N
    # print "    MODE   = ", MODE
    
    eps = 0.00001
    #declare space for samples
    x = zeros(N, dtype = float)
    x[0] = xinit

    #maximum times to expand or shrink the interval
    maxsearch=10

    #sample n points from the distribution
    #pdb.set_trace()
    for i in arange( 1, N, 1 ):
        #pick the slice level from a uniform density under the logposterior curve
        logprob_old = logdist(x[i-1],params)
        assert isnan(logprob_old) == False, 'Slice Error: logdist returned NaN'
        z = logprob_old-exponential(1.0)

        #Determine the interval
        if MODE==0:  # shrinkage on the given interval
            l=L
            r=R
        elif MODE==1: # stepping out
            c = rand()
            l=x[i-1]-c*W
            if l <= L:
                l = L+eps
            r=l+W
            if r >= R:
                r = R-eps

            logprobl = logdist( l, params)
            assert isnan(logprobl) == False, 'Slice Error: logdist returned NaN'
            j=0
            while logprobl > z and j<maxsearch:
                l-=W
                if l <= L:
                    l = L+eps
                    break
                logprobl = logdist( l, params)
                assert isnan(logprobl) == False, 'Slice Error: logdistreturned NaN'
                j=j+1

            logprobr = logdist( r, params)
            assert isnan(logprobr) == False, 'Slice Error: logdist returned NaN'
            j=0;
            while logprobr > z and j<maxsearch:
                r+=W
                if r >= R:
                    r = R-eps
                    break
                logprobr = logdist( r, params)
                assert isnan(logprobr) == False, 'Slice Error: logdistreturned NaN'
                j=j+1

        elif MODE==2: # doubling
            c = rand()
            l=x[i-1]-c*W
            if l < L+eps:
                l = L+eps
            r=l+W
            if r > R:
                r = R-eps

            logprobl = logdist( l, params)
            assert isnan(logprobl) == False, 'Slice Error: logdist returned NaN'
            logprobr = logdist( r, params)
            assert isnan(logprobr) == False, 'Slice Error: logdist returned NaN'
            j=0
            while (j<maxsearch and (logprobl > z or logprobr > z) and (abs(r-R)>2*eps or abs(l-L)>2*eps )):
                c=rand()
                j=j+1
                if (c<0.5 or R==r) and l>L:
                    l -= (r-l)
                    if l<=L:
                        l=L+eps
                    logprobl = logdist(l, params)
                    assert isnan(logprobr) == False, 'Slice Error:logdist returned NaN'
                elif (c>=0.5 or L==l) and r<R:
                    r +=  (r-l)
                    if r>=R:
                        r=R-eps
                    logprobr = logdist(r, params)
                    assert isnan(logprobr) == False, 'Slice Error:logdist returned NaN'

        #shrink until we draw a good sample
        j=0
        while j < maxsearch:
            j=j+1
            # randomly sample a new alpha on the interval
            x_new = l + rand()*(r-l)

            # compute the log posterior probability for the new alpha
            # any function proportional to the true posterior is fine
            logprob  = logdist( x_new,params)
            assert isnan(logprob) == False, 'Slice Error: logdist returned NaN'

            # Accept the sample if the log probability lies above z
            if logprob>z or abs(r-l)<eps:
                x[i] = x_new
                break
            else:
                # If not, shrink the interval
                if x_new<x[i-1]:
                    l = x_new
                else:
                    r = x_new
        # check to see if a new value was assigned.
        # if not assign the previous value and we try again.
        if x[i]==0 and j==maxsearch:
            x[i] = x[i-1]
    return x, logprob