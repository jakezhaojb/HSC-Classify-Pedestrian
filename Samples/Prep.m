function flag = Prep( datafile, resfile )
if exist(resfile,'file')
    flag = true;
    return 
end

% Note that according to be configuration (patches are sized of 5*5), we need a large
% amount of considered patches.
X_Gross = zeros(25,250*30);     %1000 images in total and 250 for each class.

for n = 1:250
    I = imread([datafile,'/',num2str(n),'.jpg']);
    for i = 3:2:125
        for j = 3:2:61                      %Cropping the image into patches, which will be used in K-SVD Dictionary Learning
            num  = ((i-3)/2)*30 + (j-3)/2+1;
            temp = I(i-2:i+2,j-2:j+2);
            X_Gross(:,num) = reshape(temp',[25,1]);             % Concatenation.
        end
    end
end
save(resfile,'X_Gross');

end

