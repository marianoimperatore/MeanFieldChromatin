## Heteropolymers Mean Field approach to predict Chromatin sites of protein binding
This framework is composed by two main scripts:
- infer_bs: PRISMR-based mean field approach to infer the best binding sites (bs) distribution on a given chromosomal region [Bianco2018];
- predict_bs: probabilistic prediction of bs in trans, given the bs classes identified by infer_bs and given the H3K27ac and RNA-seq tracks.

### Requirements
scripts have been tested for:
- pandas 1.3.4, numpy 1.21.3, scipy 1.8.0, seaborn 0.11.2, cooler 0.8.11
- python 3.9.0
- Unix environment

### Inference of the best binding sites distribution
to run the script launch the following command:

$ python infer_bs.py nod:1,30,40,4,12345 reg:GSC275B,C88bex2,5000 param:12 cfmax:0.8,1 filt:.5

- reg:GSC275B,C88bex2,5000 : the script will read the region identified as 'C88bex2' in file setInvDict.py from sample GSC275B with resolution 5kbp. This can be done either via cooler file, either via direct csv matrix file as in the presented case in folder 'data' (Hi-C file 'data/GSC275B-Arima-allReps-filtered-GSC275B+C88bex2+5000.csv' from chr12:57.66-58.33 Mbp)
- param:12 : the simulated annealing procedure (SA) will minimize the loss function using 12 monomer classes (11 bs classes + 1 class of inert polymer sites)
- filt:.5 : the Hi-C matrix will go through a Gaussian filter of 0.5 bin width
- cfmax=0.8,1 : to mitigate the presence of outliers around the diagonal the Hi-C matrix will be leveled at the 0.8 fraction of its maximum value (cfmax=0.8), or instead left unchanged (cfmax=1) [Bianco2018]
- nod:1,30,40,4,12345 : keyword to parallelize the job. 1 is the id of the single node, that varies between 1-30, 30 being the total number of parallel nodes. Every node will create a different output pandas DataFrame 'resultspd_*'. The best number of bs classes and the best regularization parameter lambda is to be evaluated from the aggregation of all files. 

In resultspd are stored the number of bs classes M, the number of bins of the Hi-C matrix N, the number of polymer beads per bin R (that usually coincides with M [Bianco2018]), the fraction alpha of the globule phase predicted, the lambda, the scaling coefficient of the monomer 'alpscal', the value of the cost function 'cost' in addition to the binding site class for each polymer bead (from 0 to M, 0 being the inert monomer class).

### Prediction of the binding sites occupancy  
In script predict_bs.py you can choose the case of interest between sample '148' and '394B' by modifying variable strSampl accordingly in the first lines. No command line options are conceived.
Plot and csv of the predicted contact matrix of the SV will be created with name 'cm+sv_pred*', along with density profiles of the predicted bs classes, the correlation matrix between tracks and classes 'classcorr', the conditional probability matrix 'condprob' and the predicted bs probability in file 'predictedbs' csv. File 'bestInference' contains the best binding sites inferred from the previous step.


### References
Bianco2018 : Bianco, S., Lupiáñez, D.G., Chiariello, A.M. et al. Polymer physics predicts the effects of structural variants on chromatin architecture. Nat Genet 50, 662–667 (2018). https://doi.org/10.1038/s41588-018-0098-8

