from rnd import pdf , cdf , rf
from scipy.optimize import minimize 
import numpy as np
import warnings

c = cdf( 20 , 10 )
d = rf( 100 , c )

def LL( x , d ) :
    return -np.sum( np.log( pdf( d , x[ 0 ] , x[ 1 ] ) ) )

def optim( LL , x0 , d , verbose = False ) :

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


    if verbose : # print results
        print( '-----OPTIM-----' )
        print( 'mu : ' 
	            + str( np.round( mu[ 0 ] , 2 ) )
	            + ' +/- '
	            + str( np.round( mu[ 1 ] , 2 ) ) )
        print( 'sigma : ' 
	            + str( np.round( sigma[ 0 ] , 2 ) )
	            + ' +/- '
	            + str( np.round( sigma[ 1 ] , 2 ) ) )

    return mu , sigma

def bootstrap( LL , x0 , d ) :

    # ------------------------------------------------------------------
    # define the Shannon entropy scoring function
    def S( dm , ds ) :

        np.seterr(divide='ignore', invalid='ignore')

        # conpute the inversion of the deltas. See point 6 in the section
        # Quantification and Statistical Analysis - Image processing and 
        # distance measurements, in Picco et al. Cell 2018
        Idm = 1/np.array( dm )
        Ids = 1/np.array( ds )

        p_m = Idm / np.sum( Idm )
        p_s = Ids / np.sum( Ids )

        return - np.sum( p_m * np.log( p_m ) - p_s * np.log( p_s ) )
    # ------------------------------------------------------------------

    # order the distance values. Important outliers are in the right tail, 
    # so it is convenient to start removing those

    dd = np.sort( d ) 
    m , s = optim( LL , x0 , dd ) 

    # storage for the incremental changes in the 
    # estimate of mu (m), sigma (s), and the 
    # Shannon entropy output (sh)
    mu = [ m ]
    sigma =  [ s ]
    dm = [] # delta mu
    ds = [] # delta sigma
    sh = [] # shannon entropy

    # start the outliers search
    search = True
    while( search ) :

        # remove  a dd
        dd = dd[ :-1]
        x0 = [ mu[ -1 ][ 0 ] , sigma[ -1 ][ 0 ] ]
        m , s = optim( LL , x0 = x0 , d = dd , verbose = False ) 

        # compute the deltas
        dm.append( np.abs( mu[ -1 ][ 0 ] - m[ 0 ] ) )
        ds.append( np.abs( sigma[ -1 ][ 0 ] - s[ 0 ] ) )
        # store the mu and sigma values
        mu.append( m )
        sigma.append( s )
        # compute and store the shannon entropy 
        sh.append( S( dm , ds ) )

        if len( dd ) < 2 * len( d ) / 3 :
            search = False
    
    print( '-----BOOTSTRAP-----' )
    print( 'max Sh: ' + str( np.nanmax( sh ) ) )
    i_max = sh.index( np.nanmax( sh ) )
    print( 'mu = ' + str( mu[ i_max ][ 0 ] ) + ' +/- ' + str( mu[ i_max ][ 1 ] ) )
    print( 'sigma = ' + str( sigma[ i_max ][ 0 ] ) + ' +/- ' + str( sigma[ i_max ][ 1 ] ) )
    print( 'min Sh: ' + str( np.nanmin( sh ) ) )
    i_min = sh.index( np.nanmin( sh ) )
    print( 'mu = ' + str( mu[ i_min ][ 0 ] ) + ' +/- ' + str( mu[ i_min ][ 1 ] ) )
    print( 'sigma = ' + str( sigma[ i_min ][ 0 ] ) + ' +/- ' + str( sigma[ i_min ][ 1 ] ) )
    
    return mu , sigma , sh , i_min , i_max
