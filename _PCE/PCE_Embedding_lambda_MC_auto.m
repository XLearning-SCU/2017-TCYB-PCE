%% -------  Principle Components Embedding for subspace learning
close all;
clear all;
clc;

% --------------------------------------------------------------------------
addpath ('../usage/');
addpath ('../data/');
addpath ('../classifers/');
addpath ('../classifers/SVM/');

% =================== loading data
CurData = 'AR_55_40_700vs700';
load (CurData);  
% ---------- data optionsameters configuration
options.nClass             = max(trainlabels);
options.nDim               = 2200;
% ---------- PCE optionsameters configuration
options.lambda             = [5:5:100]; % balance factor
options.PCARatio           = 1;   
options.k                  = zeros(size(options.lambda)); % k is calculated using lambda

% =================== loading data
% CurData = 'ExYaleB_54_48_380vs1824';
% load (CurData);  
% % ---------- data optionsameters configuration
% options.nClass             = max(trainlabels);
% options.nDim               = 2592;
% % ---------- PCE optionsameters configuration
% options.lambda             = [15:19 21:30]; % balance factor
% options.PCARatio           = 1;   
% options.k                  = zeros(size(options.lambda)); % k is calculated using lambda


% =================== loading data
% CurData = 'COIL100_64_64_500vs500';
% load (CurData);  
% % ---------- data optionsameters configuration
% options.nClass             = max(trainlabels);
% options.nDim               = 4096;
% % ---------- PCE optionsameters configuration
% options.lambda             = [1:20]; % balance factor
% options.PCARatio           = 1;   
% options.k                  = zeros(size(options.lambda)); % k is calculated using lambda

% =================== loading data
% CurData = 'MPIES4_5vs5';
% load (CurData);  
% % ---------- data optionsameters configuration
% options.nClass             = max(trainlabels);
% options.nDim               = 2050;
% % ---------- PCE optionsameters configuration
% options.lambda             = [15:30]; % balance factor
% options.PCARatio           = 1;   
% options.k                  = zeros(size(options.lambda)); % k is calculated using lambda


% =================== loading data
% CurData = 'USPS_5500_550';
% load (CurData);  
% % ---------- data optionsameters configuration
% options.nClass             = max(trainlabels);
% options.nDim               = 256;
% % ---------- PCE optionsameters configuration
% options.lambda             = [0.01:0.01:0.1]; % balance factor
% options.PCARatio           = 1;   
% options.k                  = zeros(size(options.lambda)); % k is calculated using lambda

% =================== loading data
% CurData = 'ExYaleB_54_48_SCRate50_GaussianRate30_1102vs1102';
% load (CurData);  
% % ---------- data optionsameters configuration
% options.nClass             = max(trainlabels);
% options.nDim               = 2592;
% % ---------- PCE optionsameters configuration
% options.lambda             = [5]; % balance factor
% options.PCARatio           = 1;   
% options.k                  = zeros(size(options.lambda)); % k is calculated using lambda


% =================== loading data
% CurData = 'ExYaleB_54_48_SCRate50_PixCorruptionRate30_1102vs1102';
% load (CurData);  
% % ---------- data optionsameters configuration
% options.nClass             = max(trainlabels);
% options.nDim               = 2592;
% % ---------- PCE optionsameters configuration
% options.lambda             = [5:20]; % balance factor
% options.PCARatio           = 1;   
% options.k                  = zeros(size(options.lambda)); % k is calculated using lambda

% % % =================== loading data
% CurData = 'AR_scarve_permute_600vs600';
% load (CurData);  
% % ---------- data optionsameters configuration
% options.nClass             = max(trainlabels);
% options.nDim               = 2200;
% % ---------- PCE optionsameters configuration
% options.lambda             = [5:5:100]; % balance factor
% options.PCARatio           = 1;   
% options.k                  = zeros(size(options.lambda)); % k is calculated using lambda


% ---------- preprocess the data
[tr_dat tt_dat trls ttls] = Preprocess(NewTrain_DAT, NewTest_DAT, trainlabels, testlabels, options);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels;
options.gnd = trls;
options.TRnum=length(trls);

options.NameStr = ['PCE_SL_' CurData '_Class' num2str(options.nClass) '_PCAdim' num2str(options.nDim)  '_lambda#' num2str(length(options.lambda)) '_MC_auto'];

for i = 1:length(options.lambda)
    fprintf([' * Running the experiment when lambda = ' num2str(options.lambda(i)) ' ---------\n ']);     
    % Calculate Principal Component and estimate the value of k based on lambda 
    tic;
    [U S V] = svd(tr_dat,'econ');
    options.k(i) = solve_k(S,options.lambda(i));    
                              = V(:,1:options.k(i))*V(:,1:options.k(i))';
    CKSym = CKSym + CKSym' - CKSym*CKSym';
    PC_tElapsed(i)=toc;
    fprintf([' | The time cost for getting Principal coef matrix is ' num2str(PC_tElapsed(i)) ' seconds, where | lambda = ' num2str(options.lambda(i))  '\n']);
    options.ReducedDim = options.k(i);    
    % performing dimension reduction
    tic;
    [eigvector, eigvalue] = LGE(CKSym, [], options, tr_dat');   
    tr_y = eigvector(:,1:options.ReducedDim)'*tr_dat;
    tt_y = eigvector(:,1:options.ReducedDim)'*tt_dat;
    DR_time(i) = toc;
    % performing classification
    
    tic
    SRC_rec(i) = SRC(tr_y, tt_y, trls, ttls);% note that, SRC will achieve better result by enforcing CKSym = abs(CKSym). 
    SRC_time(i) = PC_tElapsed(i)+DR_time(i)+toc;
    
    
    tic;
    SVM_rec(i) = SVM(tr_y, tt_y, trls, ttls);
    SVM_time(i) = PC_tElapsed(i)+DR_time(i)+toc;
    
    tic;
    KNN_rec(i) = NN(tr_y, tt_y, trls, ttls);    
    KNN_time(i) = PC_tElapsed(i)+DR_time(i)+toc;
   
end;

clear fid tElapsed fid ans Predict_label kk trls ttls tr_dat DAT options.gnd;
clear LapKernel SingVals i j pos t_accuracy t_nmi k tt_dat labels tmp_iter tmp_x;
clear CKSym Predict_label coef t_1nn_ac t_svm_ac;
clear L S U V Z k r K DAT order;
clear eigvector eigvalue options.gnd tr_y tt_y;
save (options.NameStr);

AnalyzeResult_MC;