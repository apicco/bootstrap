Bootstrap method used in the Picco et al. Cell, 2018

*rnd.py* contains functions for generating random data obbeying the pdf from Churchman et al. 2006

*lle.py* max likelihood estimate of the parameters of pdf and bootstrap method. Note that the bootstrap method can be largely improved. Removing distances from the largest ones is a big shortcut. Ideally, one would need to better sample the dataset drawing random subsets of it. 

In addition, it is assumed that the contamination of the dataset is minor up to only about 20% of the data being false. 

*test.py* a test scrip to test the algorithm. comment 'np.random.seed( 1 ) to test on different, random, datasets

