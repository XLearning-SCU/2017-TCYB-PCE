%% -------  Principle Components Embedding for subspace learning
close all;
clear all;
clc;

%% --------------------------------------------------------------------------
addpath ('../usage/');
addpath ('../data/');
addpath ('../classifers/');
addpath ('../classifers/SVM/');


%% =================== loading data
CurData = 'ExYaleB_54_48';
load (CurData);  
% ---------- parameter configuration
options.nClass             = max(labels);
options.nDim               = size(DAT,1);
options.trnum              = 40 * length(unique(labels));% x samples per subject
options.lambda             = [5 50 5]; % balance factor, corresponding to the classifiers 
options.ttnum              = length(labels) - options.trnum;

% %% =================== loading data
% CurData = 'ExYaleB_54_48_SCRate50_GaussianRate10';
% load (CurData);  
% % ---------- parameter configuration
% options.nClass             = max(labels);
% options.nDim               = size(DAT,1);
% options.trnum              = 29 * length(unique(labels));% x samples per subject
% options.lambda             = [25 55 10]; % balance factor, corresponding to the classifiers 
% options.ttnum              = length(labels) - options.trnum;

% %% =================== loading data
% CurData = 'ExYaleB_54_48_SCRate50_GaussianRate30';
% load (CurData);  
% % ---------- parameter configuration
% options.nClass             = max(labels);
% options.nDim               = size(DAT,1);
% options.trnum              = 29 * length(unique(labels));% x samples per subject
% options.lambda             = [5 10 5]; % balance factor, corresponding to the classifiers 
% options.ttnum              = length(labels) - options.trnum;

% %% =================== loading data
% CurData = 'ExYaleB_54_48_SCRate50_PixCorruptionRate10';
% load (CurData);  
% % ---------- parameter configuration
% options.nClass             = max(labels);
% options.nDim               = size(DAT,1);
% options.trnum              = 29 * length(unique(labels));% x samples per subject
% options.lambda             = [100 65 5]; % balance factor, corresponding to the classifiers 
% options.ttnum              = length(labels) - options.trnum;

% %% =================== loading data
% CurData = 'ExYaleB_54_48_SCRate50_PixCorruptionRate30';
% load (CurData);  
% % ---------- parameter configuration
% options.nClass             = max(labels);
% options.nDim               = size(DAT,1);
% options.trnum              = 29 * length(unique(labels));% x samples per subject
% options.lambda             = [90 35 10]; % balance factor, corresponding to the classifiers 
% options.ttnum              = length(labels) - options.trnum;

% %% =================== loading data
% CurData = 'AR_scarve';
% load (CurData);  
% % ---------- parameter configuration
% options.nClass             = max(labels);
% options.nDim               = size(DAT,1);
% options.trnum              = 6 * length(unique(labels));% x samples per subject
% options.lambda             = [65 65 55]; % balance factor, corresponding to the classifiers 
% options.ttnum              = length(labels) - options.trnum;


% %% =================== loading data
% CurData = 'AR_glass';
% load (CurData);  
% % ---------- parameter configuration
% options.nClass             = max(labels);
% options.nDim               = size(DAT,1);
% options.trnum              = 6 * length(unique(labels));% x samples per subject
% options.lambda             = [50 95 70]; % the tuned lambdas corresponding to the used three classifiers.
% options.ttnum              = length(labels) - options.trnum;


% ---------- experimental parameter configuration
options.nRep               = 10; % number to repeat experiment 
options.PCARatio           = 1;  % not travial SVs are truncated unlike PCA 
options.k                  = zeros(size(options.lambda)); % k is calculated using lambda
options.NameStr = ['PCE_SL_' CurData '_' num2str(options.trnum) 'vs' num2str(options.ttnum) '_Class' num2str(options.nClass) '_PCAdim' num2str(options.nDim)  '_Rep#' num2str(options.nRep) '_auto'];
%% ---------- preprocess the data
DAT        =   double(DAT(:,labels<=options.nClass));
DAT        =   DAT./( repmat(sqrt(sum(DAT.*DAT)), [size(DAT, 1),1]) );
labels     =   labels(labels<=options.nClass);

for i = 1:options.nRep
    ind = randperm(length(labels));
    tr_dat = DAT(:,ind(1:options.trnum));
    trls   = labels(ind(1:options.trnum));
    tt_dat = DAT(:,ind(1+options.trnum:end));
    ttls   = labels(ind(1+options.trnum:end));
    options.gnd = trls;
    clear ind;
 
    %% ---------SRC
    ClassifierInd = 1;
    fprintf(' * with SRC, Running the %d-th experiment * \n', i);     
    % Calculate Principal Component and estimate the value of k based on lambda 
    tic;
    [U S V] = svd(tr_dat,'econ');
    options.k(ClassifierInd) = solve_k(S,options.lambda(ClassifierInd));    
    CKSym = V(:,1:options.k(ClassifierInd))*V(:,1:options.k(ClassifierInd))';
    CKSym = CKSym + CKSym' - CKSym*CKSym';
    SRC_PC_tElapsed(i)=toc;
    
    % performing dimension reduction
    tic;
    options.ReducedDim = options.k(ClassifierInd);
    [eigvector, eigvalue] = LGE(CKSym, [], options, tr_dat');
    tr_y = eigvector(:,1:options.ReducedDim)'*tr_dat;
    tt_y = eigvector(:,1:options.ReducedDim)'*tt_dat;
    SRC_DR_time(i) = toc;
    % performing classification
    tic;
    SRC_rec(i)  = SRC(tr_y, tt_y, trls, ttls);% note that, SRC will achieve better result by enforcing CKSym = abs(CKSym). 
    SRC_time(i) = SRC_PC_tElapsed(i)+SRC_DR_time(i)+toc;

      
    
    %% --------- SVM
    ClassifierInd = 2;
    fprintf(' * with SVM, Running the %d-th experiment * \n', i);     
    % Calculate Principal Component and estimate the value of k based on lambda 
    tic;
    [U S V] = svd(tr_dat,'econ');
    options.k(ClassifierInd) = solve_k(S,options.lambda(ClassifierInd));    
    CKSym = V(:,1:options.k(ClassifierInd))*V(:,1:options.k(ClassifierInd))';
    CKSym = CKSym + CKSym' - CKSym*CKSym';
    SVM_PC_tElapsed(i)=toc;
    
    % performing dimension reduction
    tic;
    options.ReducedDim = options.k(ClassifierInd);
    [eigvector, eigvalue] = LGE(CKSym, [], options, tr_dat');
    tr_y = eigvector(:,1:options.ReducedDim)'*tr_dat;
    tt_y = eigvector(:,1:options.ReducedDim)'*tt_dat;
    SVM_DR_time(i) = toc;
    % performing classification    
    tic;
    SVM_rec(i) = SVM(tr_y, tt_y, trls, ttls);
    SVM_time(i) = SVM_PC_tElapsed(i)+SVM_DR_time(i)+toc;

    %% --------- NN
    ClassifierInd = 3;
    fprintf(' * with KNN, Running the %d-th experiment * \n', i);     
    % Calculate Principal Component and estimate the value of k based on lambda 
    tic;
    [U S V] = svd(tr_dat,'econ');
    options.k(ClassifierInd) = solve_k(S,options.lambda(ClassifierInd));    
    CKSym = V(:,1:options.k(ClassifierInd))*V(:,1:options.k(ClassifierInd))';
    CKSym = CKSym + CKSym' - CKSym*CKSym';
    KNN_PC_tElapsed(i)=toc;
    
    % performing dimension reduction
    tic;
    options.ReducedDim = options.k(ClassifierInd);
    [eigvector, eigvalue] = LGE(CKSym, [], options, tr_dat');
    tr_y = eigvector(:,1:options.ReducedDim)'*tr_dat;
    tt_y = eigvector(:,1:options.ReducedDim)'*tt_dat;
    KNN_DR_time(i) = toc;
    % performing classification    
    tic;
    KNN_rec(i) = NN(tr_y, tt_y, trls, ttls);    
    KNN_time(i) = KNN_PC_tElapsed(i)+KNN_DR_time(i)+toc;
   
end;

clear fid tElapsed fid ans Predict_label kk trls ttls tr_dat DAT options.gnd;
clear LapKernel SingVals i j pos t_accuracy t_nmi k tt_dat labels tmp_iter tmp_x;
clear CKSym Predict_label coef t_1nn_ac t_svm_ac;
clear L S U V Z k r K DAT order;
clear eigvector eigvalue options.gnd tr_y tt_y options.gnd;
save (options.NameStr);

AnalyzeResult_Rep;