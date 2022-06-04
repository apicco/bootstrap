Bootstrap method used in the Picco et al. Cell, 2018

*rnd.py* contains functions for generating random data obbeying the pdf from Churchman et al. 2006

*lle.py* max likelihood estimate of the parameters of pdf and bootstrap method. Note that the bootstrap method can be (largely?) improved and slightly differs from the R code:
- the optimiser is used in its default configuation, a better method choice could improve the optimization of the MLE. There is no guarantee that 'minimize' outputs the same numerical results as 'mle2' in R.
- distances are sorted to ease the focus of the bootstrap on distance measurements that are large, which are those most likely to hide outliers

It is assumed that the contamination of the dataset is minor up to only about 20% of the data being false. 
The software has been tested with contaminations in this range only. Larger contaminations (i.e. poorer images) might lower the softwer performance and might require more powerful algorithms to clean the data.

*test.py* a test scrip to test the algorithm. comment 'np.random.seed( 1 ) to test on different, random, datasets

