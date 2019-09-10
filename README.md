# 2017-TCYB-PCE

The databases and codes were used to reproduce the results reported in our submission. 

% =========================================================================

1. If you would like to reproduce the result, please assign the parameters with the value reported in the submission, otherwise, specify some possible values for the parameters.

3. The tuned parameter in our submission is determined using a validation data. More specifically, we obtain 11 paritions of the data set (e.g., AR) consisting of traning and testing sets. We randomly use one parition to find optimal parameters for all the methods, and then used the tuned parameters to perform 10 tests on the remaining 10 paritions. 

3. The codes were produced in MATLAB2014a 64 bit on a MACBOOK.

### Data
1. AR_55_40_700vs700: a validate data set to tuned parameter. 'PCE_Embedding_lambda_MC_auto.m' requires it;
2. All data sets except AR_55_40_700vs700.mat are full data set, which are the inputs of 'PCE_Embedding_lambda_Repeat_auto.m'.

### _PCE
The implementation of our method 

1. PCE_Embedding_lambda_MC_auto.m: to perform PCE on the validate data set;
2. PCE_Embedding_lambda_Repeat_auto.m: to perform multiple tests on a data set with the tuned lambda;
3. solve_k.m: automatically determine the feature dimension;
4. AnalyzeResult_MC.m: As the name implies, run it to analyze the result of 'PCE_Embedding_lambda_MC_auto' with different parameter combinations;
5. AnalyzeResult_Rep: to calculate the $mean\pm std$ based on 'PCE_Embedding_lambda_Repeat_auto';

### usages
Some codes sharing by the tested methods.
1. Eigenface_f.m: reduce the dim. of the data using PCA;2
2. EuDist2.m: calculate the pairwise distance among data points, for LPP;
3. LGE.m: embedding function, provided by He et al.;
4. mySVD.m: perform SVD;
5. Preprocess.m: preprocess data if the input dim is larger than target dim. In the paper, no preprocess except normalization.  

### classifers
The used three classifiers, i.e, SRC, SVM with linear kernel, and the nearest neighbor classifier.
Note that:
1. if you want to perform experiments on Windows instead of Mac OS, pls use the files under the fold of SVM_win;
2. if you use 32 bits machine, pls recomplie SVM on you machine.

## Citation
* Xi Peng, Jiwen Lu, Zhang Yi, and Yan Rui, Automatic Subspace Learning via Principal Coefficients Embedding, IEEE Trans Cybernetics (TCYB), vol. 47, no. 11, pp. 3583-3596, Nov. 2017. DOI:10.1109/TCYB.2016.2572306.

* @article{Peng2017:Automatic_full,  
  author={Xi Peng and Jiwen Lu and Zhang Yi and Yan Rui},   
  journal={IEEE Transactions on Cybernetics},   
  title={Automatic Subspace Learning via Principal Coefficients Embedding},   
  year={2017},   
  volume={47},   
  number={11},   
  pages={3583--3596},   
  doi={DOI:10.1109/TCYB.2016.2572306},   
  ISSN={2168-2267},   
  month={Nov.},}
