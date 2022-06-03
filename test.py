from rnd import pdf , cdf , rf
from lle import bootstrap , LL , optim
from matplotlib import pyplot as plt
import numpy as np

np.random.seed( 1 ) 
mu0 = 20
sigma0 = 10
x0 = [ 18 , 12 ]
c = cdf( mu0 , sigma0 )
# generate random data with and without contamination
d0 , d = rf( 100 , c , 20 )
# compute the estimate and verbose the result of the MLE on the data without contamination
mu_true , sigma_true = optim( LL , x0 = x0 , d = d0 , verbose = True )

# compute the MLE on the data with contamination
mu , sigma , sh , i_min , i_max = bootstrap( LL , x0 = x0 , d = d )

# output a summary plot
f = plt.figure()
plt.hist( d0 , density = True , color = 'green' , alpha = 0.5 , label = 'Uncontaminated data' )
plt.hist( d , density = True  , color = 'red' , alpha = 0.5 , label = 'Contaminated data' )
plt.plot( c[ 0 ] , pdf( c[ 0 ] , mu0 , sigma0 ) , color = 'black' , 
    label = 'True values: $\mu=$' + str( round( mu0 , 2 ) ) + ' nm, $\sigma=$' + str( round( sigma0 , 2 ) ) + ' nm' )
plt.plot( c[ 0 ] , pdf( c[ 0 ] , mu_true[ 0 ] , sigma_true[ 0 ] ) , color = 'green' ,
    label = 'Estimate from uncontaminated data:\n$\mu=$' 
    + str( round( mu_true[ 0 ] , 2 ) ) + '$\pm$' + str( round( mu_true[ 1 ] , 2 ) ) + 'nm, $\sigma=$' 
    + str( round( sigma_true[ 0 ] , 2 ) ) + '$\pm$' + str( round( sigma_true[ 1 ] , 2 ) ) + 'nm' )
plt.plot( c[ 0 ] , pdf( c[ 0 ] , mu[ i_min ][ 0 ] , sigma[ i_min ][ 0 ] ) , color = 'red' ,
    label = 'Estimate from contaminated data,\nwith bootstrap:\n$\mu=$' 
    + str( round( mu[ i_min ][ 0 ] , 2 ) ) + '$\pm$' + str( round( mu[ i_min ][ 1 ] , 2 ) ) + 'nm, $\sigma=$' 
    + str( round( sigma[ i_min ][ 0 ] , 2 ) ) + '$\pm$' + str( round( sigma[ i_min ][ 1 ] , 2 ) ) + 'nm' )
plt.plot( c[ 0 ] , pdf( c[ 0 ] , mu[ i_max ][ 0 ] , sigma[ i_max ][ 0 ] ) , color = 'red' , ls = 'dashed' ,
    label = 'Estimate from contaminated data,\nwithout bootstrap:\n$\mu=$' 
    + str( round( mu[ i_max ][ 0 ] , 2 ) ) + '$\pm$' + str( round( mu[ i_max ][ 1 ] , 2 ) ) + 'nm, $\sigma=$' 
    + str( round( sigma[ i_max ][ 0 ] , 2 ) ) + '$\pm$' + str( round( sigma[ i_max ][ 1 ] , 2 ) ) + 'nm' )

plt.xlabel('$\mu$ (nm)' )
plt.ylabel( 'Density' )
plt.legend()

inset = f.add_axes( [ 0.5 , 0.205 , 0.35 , 0.20 ] )
inset.plot( [ mu[i][ 0 ] for i in range( len( sh ) ) ] , sh )
inset.set_xlabel( '$\mu$ (nm)' )
inset.set_ylabel( 'Entropy (a.u.)' )
plt.savefig( "test.pdf" )

