import numpy as np
from scipy.special import iv as besselI

# probability density function for distances in 2D. Eq.(4) in Churcham et al. 2006 
def pdf( l , mu , sigma ) :

    # the equation
    def f( r , mu , sigma ) :
   
        return ( r / sigma ** 2 ) * np.exp( - ( mu ** 2 + r ** 2 ) / ( 2 * sigma ** 2 ) ) * besselI( 0 , r  * mu / sigma ** 2 )

    # output in function of the input length
    if np.size( l ) > 1 :
        return [ f( x , mu , sigma ) for x in l ]
    elif np.size( l ) == 1 :
        return f( l , mu , sigma ) 

# comulative density function approx of pdf
def cdf( mu , sigma , dx = 0.01 , x0 = 0 , x1 = 2*1E2 ) : 

    l = [ i for i in np.arange( x0 , x1 , dx ) ]
    return [ l , dx * np.cumsum( pdf( l , mu , sigma ) ) ]

# random values from pdf, n values, given the cdf c , adding noise datapoint
# of noise, which are datapoint not obbeying the pdf. These datapoints are
# those that should be removed by the bootstrapping
def rf( n , c , noise = 0 , noise_mean = np.nan , noise_std = np.nan ) :

    # define the vectors to store the distances
    x = np.zeros( n  )
    xn = np.zeros( n + noise )

    # generate n random values to be used with the cdf to 
    # determine semi-random distance values obbeying the pdf
    y = np.random.rand( n )

    for i in range( n ) :

        # find the cdf value c that is the closest to y
        tmp = min( [ j for j in c[ 1 ] if j > y[ i ] ] )
        # map it to its corresponding distance, and store it in x
        x[ i ] = c[ 0 ][ c[ 1 ].tolist().index( tmp ) ]

    # if noise mean and std are nan, create default values
    if noise_mean != noise_mean : noise_mean = max( x )
    if noise_std != noise_std : noise_std = noise_mean / 3

    xn[ : i + 1 ] = x
    xn[ i + 1 : ] = np.abs( np.random.normal( 
            loc = noise_mean , 
            scale = noise_std , 
            size = noise ) ) # abs is to avoid possible neg values. Neg distances do not exist

    # shuffle and return xn so that the noisy 
    # values are not only at the end of the vector
    np.random.shuffle( xn )
    return x , xn
