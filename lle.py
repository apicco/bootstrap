from rnd import pdf , cdf , rf
from scipy.optimize import minimize 
import numpy as np
import warnings

c = cdf( 20 , 10 )
d = rf( 100 , c )

def LL( x , d ) :
    # IMPORTANT NOTE: LL is defined as minus the log likelihood because its values
    # will be optimised through a minimisation process. Hence, the values that maximise
    # the log likelihood are those that minimise LL. This note will be important
    # in the bootstrap method, as the outliers rejected are those whore rejection
    # increases the log likelihood, thus decreases LL
    pp = pdf( d , x[ 0 ] , x[ 1 ] )
    return -np.nansum( np.log( pp ) )

# define the Shannon entropy scoring function
def S( mus , sigmas ) :

    N = len( mus )
    M = len( sigmas )

    # convert list to numpy array
    mus = np.array( [ mus[ i ][ 0 ] for i in range( N ) ] )
    sigmas = np.array( [ sigmas[ i ][ 0 ] for i in range( M ) ] ) 

    # conpute the inversion of the deltas. See point 6 in the section
    # Quantification and Statistical Analysis - Image processing and 
    # distance measurements, in Picco et al. Cell 2018
    np.seterr(divide='ignore', invalid='ignore')
    Im = 1/np.abs( mus[ 1: ] - mus[ :-1 ] )
    Is = 1/np.abs( sigmas[ 1: ] - sigmas[ :-1 ] )

    p_m = Im / np.nansum( Im )
    p_s = Is / np.nansum( Is )

    return - p_m * np.log( p_m ) - p_s * np.log( p_s ) 


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

def bootstrap( LL , x0 , d , cutoff = np.nan ) :

    d.sort()    # sorting of the distances is important, because it will ease the
                # min LL analysis by focusing only on the LL values past the LL max
                # thus on those values coming from the tail of the distribution
                # (i.e. the largest distances)
    if cutoff != cutoff : cutoff = 2 * len( d ) / 3 # 5 in the R code, but I think it is safe to assume 
                                                    # that more than half of the data area good

    # order the distance values. Important outliers are in the right tail, 
    # so it is convenient to start removing those

    m , s = optim( LL , x0 , d ) 

    # storage for the incremental changes in the 
    # estimate of mu (m), sigma (s), and the 
    # Shannon entropy output (sh)
    mu = [ m ]
    sigma =  [ s ]
    dd = [ d ]

    # start the outliers search
    m_tmp = x0[ 0 ] # the m and s for the LL to determine which
    s_tmp = x0[ 1 ] # d is 'more likely' do be rejected
    search = True
    while( search ) :

        # the number of distance measurements left
        n = len( dd[ -1 ] )
        # storage vector for the LL estimates when removing a distance
        l = []
        # storage vector for the distance dataset with each removed distance
        dtmp = []
        for i in range( n ) :
           
            dtmp.append( [ dd[ -1 ][ j ] for j in range( n ) if j != i ] )
            l.append( LL( [ m_tmp , s_tmp ] , dtmp[ i ] ) )

        # keep the d that is 'more likely' to belong to the dataset (i.e. max likelihood )
        # NOTE that LL has been defined with a minus for optin to search for a minimum. Thus
        # here we need to search for a max instead of a minimum, and for a minimum, instead of a max
        l_max = l.index( max( l ) ) # that is the index of the value most pertinent to the pdf, whose
                                    # removal maximises the LL ( i.e. minimises the log likelihood)
        ll = l[ l_max : ]
        i_sel = l_max + ll.index( min( ll ) )

        new_dataset = dtmp[ i_sel ]
        # compute a new optimisation on the dataset without the 'less likely' distance 
        m , s = optim( LL , x0 = x0 , d = new_dataset , verbose = False ) 

        # store the mu and sigma values
        mu.append( m )
        sigma.append( s )
        dd.append( new_dataset )
        
        if len( new_dataset ) < cutoff :
            search = False
        else :
            m_tmp = m[ 0 ]
            s_tmp = s[ 0 ]
        
    # compute and store the shannon entropy 
    sh =  S( mu , sigma ).tolist()
    
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
