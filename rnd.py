import numpy as np
from scipy.special import iv as besselI
from matplotlib import pyplot as plt

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

# random values from pdf, n values, given the cdf c
def rf( n , c ) :

    y = np.random.rand( n )
    x = np.zeros( n )

    for i in range( n ):

        # find the cdf value c that is the closest to y
        tmp = min( [ j for j in c[ 1 ] if j > y[ i ] ] )
        x[ i ] = c[ 0 ][ c[ 1 ].tolist().index( tmp ) ]

    return x

def test() :
	
    c = cdf( 20 , 20 )
	d = rf( 100 , c )
	
	
	f = plt.figure()
	plt.hist( d , density = True )
	plt.plot( c[ 0 ] , pdf( c[ 0 ] , 20 , 20 ) )
	plt.savefig( "test.pdf" )
