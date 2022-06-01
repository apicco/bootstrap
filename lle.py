from rnd import pdf , cdf , rf
from scipy.optimize import minimize 
import numpy as np

c = cdf( 20 , 10 )
d = rf( 100 , c )

def LL( x , d ) :
    return -np.sum( np.log( pdf( d , x[ 0 ] , x[ 1 ] ) ) )

def optim( LL , x0 , d ) :

    opt = minimize( LL , x0 , args = d , hess = True )
    x = opt['x']
    se =  np.sqrt( np.diagonal( opt['hess_inv'] ) )

    mu = [ x[ 0 ] , se[ 0 ] ]
    sigma = [ x[ 1 ] , se[ 1 ] ]

    return mu , sigma

mu , sigma = optim( LL , x0 = [ 10 , 10 ] , d = d )
