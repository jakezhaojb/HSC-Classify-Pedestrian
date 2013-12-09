function Prep_tr_te(tr_sample_num)
%For training
if ~exist('Sample/sample_tr.mat','file')
    tr_num = tr_sample_num * 30;
    X = zeros(25,4*tr_num);
    la = 1:1:tr_num;
    for i = 1:4
        load(['Sample/s_',num2str(i),'.mat'],'X_Gross');
        X(:,(i-1)*tr_num+1:i*tr_num) = X_Gross(:,la);
    end
    save('Sample/sample_tr.mat','X')
end

%For testing
if ~exist('Sample/sample_te.mat','file')
    clearvars -except tr_sample_num
    tr_num = tr_sample_num * 30;
    te_num = 250*30-tr_num;
    X = zeros(25,4*te_num);
    la = tr_num+1:1:30*250;
    for i = 1:4
        load(['Sample/s_',num2str(i),'.mat'],'X_Gross');
        X(:,(i-1)*te_num+1:i*te_num) = X_Gross(:,la);
    end
    save('Sample/sample_te.mat','X')
end