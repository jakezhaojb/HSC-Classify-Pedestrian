function flag = Dic_Learn_KSVD(datafile, resfile)
if exist(resfile,'file')
    flag = true;
    return
end
if ~exist(datafile,'file')
    flag = false;
    return
end
% Learning begins
load(datafile,'X');
noIt = 2000;            
D = randn(25,150);
op = struct('tnz',5, 'verbose', 2);     
for it = 1:noIt
    %clc
    W = sparseapprox(X, D, 'OMP', op);
    R = X - D*W;
    for k=1:150     % Parallel is prefered!!
        I = find(W(k,:));           
        Ri = R(:,I) + D(:,k)*W(k,I);        
        [U,S,V] = svds(Ri,1,'L');
        % U is normalized
        D(:,k) = U;
        W(k,I) = S*V';
        R(:,I) = Ri - D(:,k)*W(k,I);
    end    
end
dic_first.dic = D;
dic_first.patchsize = 5;
dic_first.dicsize = 150;
save(resfile,'dic_first');
flag = true;

