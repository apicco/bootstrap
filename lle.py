from rnd import pdf , cdf , rf
from scipy.optimize import minimize 
import numpy as np
import warnings

c = cdf( 20 , 10 )
d = rf( 100 , c )

def LL( x , d ) :
    return -np.sum( np.log( pdf( d , x[ 0 ] , x[ 1 ] ) ) )

def optim( LL , x0 , d ) :

    # compute the minimization catching warnings for the initial
    # method choice that might not compel with the hess = True
    # option
    with warnings.catch_warnings() :
        warnings.simplefilter( 'ignore' )
        opt = minimize( LL , x0 , args = d , hess = True )

    # extract the parameter estimate and compute their standard
    # error estimate
    x = opt['x']
    se =  np.sqrt( np.diagonal( opt['hess_inv'] ) )

    # formatting and output
    mu = [ x[ 0 ] , se[ 0 ] ]
    sigma = [ x[ 1 ] , se[ 1 ] ]

    print( 'mu : ' 
            + str( np.round( mu[ 0 ] , 2 ) )
            + ' +/- '
            + str( np.round( mu[ 1 ] , 2 ) ) )
    print( 'sigma : ' 
            + str( np.round( sigma[ 0 ] , 2 ) )
            + ' +/- '
            + str( np.round( sigma[ 1 ] , 2 ) ) )

    return mu , sigma

mu , sigma = optim( LL , x0 = [ 10 , 10 ] , d = d )
