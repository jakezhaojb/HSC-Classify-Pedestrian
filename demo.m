% This is a QUICK validation of Histogram of Sparse Coding for Object Detection, for Pedestrian Classifications.

% I propose to classify the pedestrians based on their postures, as well as their walking directions,
% which will lead the Pedestrian Detection System more intelligent.

% Designed by Junbo Zhao, Wuhan University, 2013/10/26
% Licensed by Wuhan University and NO COMMERCIAL USAGE, thanks.

% This program aims at classifying pedestrians into classes: Front, Back, Left, Right.
% The further study would focus on increasing the number of classes.

% Some related ideas could be seen in my previously published paper "Features Fusion with Adaptive
% Weights for Pedestrian Classification", in IEEE International Conference on Control, Automation
% and Systems (ICCAS), Oct. 20-23, 2013, Gwangju, Korea.

% Note that in this framework, we use 1000 images in total, which are
% cropped from INRIA Dataset. Specifically, 880 images are used as
% training data and the rest 120 images are put into use in the testing
% phase.

% In addition, I configure the Dictionary SIZE to 150 and Patch SIZE to
% 5*5

clc
clear

%Preparation for KSVD dic_learning
assert(isdir('Samples'))
addpath(genpath('Samples'))
if ~exist('Samples/sample.mat','file')
    Prep('Samples/back','Sample/s_1.mat');              %Preparing for the training data of four pedestrian classes.
    Prep('Samples/front','Sample/s_2.mat');
    Prep('Samples/left','Sample/s_3.mat');
    Prep('Samples/right','Sample/s_4.mat');
end
Prep_tr_te(220);         % In our framework, 880 images can be used in the training phase and thus 220 for each class.

% Learning the Dictionary, using K-SVD
% Bear in mind that this may take some time to finish.
if dic_learn_ksvd('Samples/sample_tr_trun.mat','Samples/Dic.mat')
    disp('The dictionary learning by KSVD is Ok')
else
    disp('The dictionary Learning by KVD is failed.')
    pause;
end


clear 
% Histograms of Sparse Coding model
if ~exist('f.mat','file')
    % To speed up the K-SVD Learning, we only use 12600 samples.
    % It is recommended that the number you use for K-SVD should 
    % not be less than our model's.
    f_gross = zeros(1000,12600);
    load('Samples/Dic.mat','dic_first');
    encoder_gray=dic_first;
    encoder_gray.type='gray';
    encoder_gray.blocksize=8;
    encoder_gray.numblock=1;
    encoder_gray.sparsity=1;
    encoder_gray.power_trans=0.25;
    encoder_gray.pad_zero=0;
    encoder_gray.threshold=0.001;
    encoder_gray.norm=2;
    
    % HSC 
    dir{1} = 'Samples/back/';
    dir{2} = 'Samples/front/';
    dir{3} = 'Samples/left/';
    dir{4} = 'Samples/right/';
    for i =1:4
        for j = 1:250
            I =imread(strcat(dir{i},[num2str(j),'.jpg']));
            f = features_hsc_1(double(I),8,encoder_gray);  %Histograms of Sparse Coding Computing
            temp = reshape(f,[84,150]);
            temp = reshape(temp,[1,12600]);
            f_gross((i-1)*250+j,:) = temp;
            disp(['Finishing ',num2str((i-1)*250+j),' th image'])
        end
    end
    save('f.mat','f_gross');
else
    load('f.mat','f_gross')
end

% Note that we exploit Lib-SVM as our classifier.
% You need to download this tool on:
% http://www.csie.ntu.edu.tw/~cjlin/libsvm/
% and follows the instructions closely.

% SVM and Evaluation
tr_num = [1:1:220,251:1:470,501:1:720,751:1:970];
f_tr = f_gross(tr_num,:);
te_num = setdiff([1:1:1000],tr_num);
f_te = f_gross(te_num,:);
f_tr_label = [ones(220,1);2*ones(220,1);3*ones(220,1);4*ones(220,1)];
f_te_label = [ones(30,1);2*ones(30,1);3*ones(30,1);4*ones(30,1)];
if ~exist('SVM_T.mat','file')
    options = optimset('maxiter', 2000, 'largescale','off');
    SVM_T = svmtrain(f_tr_label,f_tr, '-t 0');
    save('svmtranining.mat','SVM_T');
else
    load('svmtranining.mat','SVM_T');
end
res = svmpredict(f_te_label,f_te,SVM_T);
save('Results.mat','res');
%comparison goes here
for i = 1:4
    rate(i) = length(find(res(30*(i-1)+1:30*i)==f_te_label(30*(i-1)+1:30*i)))/30;
end
save('Results.mat','rate','-append');
disp(' ')
disp('Results:')
disp(rate)

%ROC curve goes here
disp('ROC curves are to be updated');
