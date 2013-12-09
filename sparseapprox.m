function varargout = sparseapprox(X, D, met, varargin)
% sparseapprox     Returns coefficients in a sparse approximation of X.
%                  Several methods for sparse approximation may be used, 
% some implemented in this m-file and others depend on external parts.
% For the methods (3) and (4) below, the corresponding packages
% should be installed on your computer and be available from Matlab.
%
% The coefficients or weights, W, are usually (but not always) sparse, 
% i.e. number of non-zero coefficients below a limit,
% some methods use the 1-norm but these are not much tested.
% 
% The reconstrucion or approximation of X is (D*W).
% The approximation error is R = X - D*W;
% The Signal to Noise Ratio is snr = 10*log10(var(X(:))/var(R(:)));
%
%   Use of function:
%   ----------------
%   W = SPARSEAPPROX(X, D, met, 'option',value, ...)
%     W is coefficient matrix, size KxL
%     X is data, a matrix of size NxL, (a column vector when L=1)
%     D is the dictionary, size NxK
%     met is a char string for the different methods, see below
%     Additional arguments may be given as pairs: 'option',value, ...
%     (or options may be given in one (or more) struct or cell-array)
%     These options may be used for the different methods.
% 
%   [W, res] = SPARSEAPPROX(X, D, met, 'option',value, ...)
%     res is a struct with additional results 
%   
%   The alternative methods are:
%   ----------------------------
%   (1) Methods that use Matlab standard functions: 'pinv', '\', 'linprog' 
%   The representation is now exact and usually not sparse unless
%   thresholding is done (see below).
% 'MOF', 'MethodOfFrames', or 'pinv'
% 'BackSlash' or '\'
% 'BP' or 'BasisPursuit' or 'linprog' 
%   (2) Methods implemented in this m-file
% 'FOCUSS' a best basis variant. Use options 'nIt' and 'pFOCUSS'.
%   When option 'lambdaFOCUSS' is given Regularized FOCUSS is used.
%   For the four methods ('pinv', 'BackSlash', 'linprog' and 'FOCUSS')
%   thresholding is done when 'tnz', 'tre' or 'tae' is given as option.
%   If 'doOP' the coefficients are set so that D*W(:,i) is an orthogonal  
%   projection onto the space spanned by used columns of D. 
% 'GMP', a Global variant of Matching Pursuit.  NOTE that option 'tnz' 
%   should be given as the total number of non-zeros in W
% 'OMP', Orthogonal Matching Pursuit 
% 'ORMP', Order Recursive Matching Pursuit 
%   (3) Methods implemented in the 'mpv2' java package (by K. Skretting)
%   see page: http://www.ux.uis.no/~karlsk/dle/index.html
%   They are all variants of Matching Pursuit
% 'javaMP', 'javaMatchingPursuit', 'javaBMP', 'javaBasicMatchingPursuit'
% 'javaOMP' or 'javaOrthogonalMatchingPursuit'
% 'javaORMP' or 'javaOrderRecursiveMatchingPursuit'
% 'javaPS' or 'javaPartialSearch'
%   (4) Methods implemented as mex-files in SPAMS (by J. Mairal)
%   These are very fast and can be used also for quite large problems.
%   see page: http://spams-devel.gforge.inria.fr/
% 'mexLasso', 'LARS', or 'LASSO'
% 'mexOMP'  NOTE: this is the same algorithm as ORMP and javaORMP!
%
%   The Options may be:
%   -------------------
%   'tnz' or 'targetNonZeros' with values as 1x1 or 1xL, gives the target
%     number of non-zero coefficients for each column vector in X
%     Default is ceil(N/2)
%   'tre' or 'targetRelativeError' with values as 1x1 or 1xL, gives the 
%     target relative error, i.e. iterations stops when ||r|| < tre*||x||.
%     If both tnz and tre is given, the iterations stops when any criterium
%     is met. Default is 1e-6.
%   'tae' or 'targetAbsoluteError' with values as 1x1 or 1xL, gives an
%     alternative way to set 'tre' on: tre = tae ./ ||x||
%     Iterations stops when ||r|| < tae. If used 'tae' overrides 'tre'.
%   'doOP' do Orthogonal Projection when thresholding, default is true
%   'nIt' or 'nofIt' or 'numberOfIterations' is used in FOCUSS, default 20.
%   'p' or 'pFOCUSS' the p-norm to use in FOCUSS. Default 0.5. 
%   'l' or 'lambdaFOCUSS' is lambda in Regularized FOCUSS, default is 0
%   'nComb' number of combinations in 'javaPS'. Default is 20.
%   'globalReDist', 'tSSE' or 'targetSSE', 'tSNR' or 'targetSNR' are 
%      undocumented options which may be used with method 'javaORMP' only. 
%   'GMPLoopSize', optional parameter used in GMP (default usually ok)
%   'paramSPAMS', optional parameter to use in mexLasso or mexOMP. If not given
%      the following will be used for mexOMP and mexLasso respectively:
%      paramSPAMS = struct( 'eps', tae.^2, 'L', int32(tnz) );
%      paramSPAMS = struct( 'mode',1, 'lambda', tae.^2, 'L', int32(tnz) );
%   'v' or 'verbose' may be 0/false (default), 1/true or 2 (very verbose).
%
%   Examples:
%   ---------
%   L = 400; N = 16; K = 32; D = randn(N,K); X = randn(N,L);
%   op = struct('targetNonZeros',5, 'verbose', 2);
%   [Wa, ra] = sparseapprox(X, D, 'pinv', op);
%   [Wc, rc] = sparseapprox(X, D, 'linprog', op); 
%   [Wd, rd] = sparseapprox(X, D, 'FOCUSS', op, 'p', 0.8, 'l', 0.4, 'nIt', 100);
%   [Wf, rf] = sparseapprox(X, D, 'javaOMP', op); 
%   [Wg, rg] = sparseapprox(X, D, 'javaORMP', op); 
%   [Wh, rh] = sparseapprox(X, D, 'javaPS', 'nComb',100, op);
%   [Wi, ri] = sparseapprox(X, D, 'GMP', 'tnz',5*L, 'v',2); 
%   [Wih, rih] = sparseapprox(X, D, 'javaPS', 'tnz',sum(Wi~=0), 'nComb',100);
%   fs = ' %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f \n';
%   fprintf('\n       pinv  linprog FOCUSS  OMP   ORMP   PS     GMP   GMP+PS \n'); 
%   fprintf(['SNR  ',fs], ra.snr,rc.snr,rd.snr,rf.snr,rg.snr,rh.snr,ri.snr,rih.snr);  
%   fprintf(['time ',fs], ra.time,rc.time,rd.time,rf.time,rg.time,rh.time,ri.time,rih.time);  

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger, Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  10.10.2009  Made function
% Ver. 1.1  04.01.2010  Add GMP method, and globalReDist in javaORMP
% Ver. 1.2  05.01.2010  added targetTotalSumSquaredError and targetSNR 
%           06.01.2010  and now it works well. 
% Ver. 1.3  26.03.2010  Add LARS and LASSO methods
% Ver. 1.4  07.04.2010  Added smoothed L_0 norm (SL0) method
% Ver. 2.0  06.04.2011  Removed some methods and tried to 'clean' this file
%                       SLO, LARS/LASSO matlab implemetations, and
%                       globalReDist after javaORMP were removed.
% Ver. 2.1  31.05.2011  globalReDist included again
% Ver. 2.2  08.08.2012  Added affine ORMP method, 
% Ver. 2.3  15.04.2013  Removed affine ORMP method (It was not as I hoped it would be)
%----------------------------------------------------------------------
%
% additional documentation:
%  - Dictionary Learning Tools: http://www.ux.uis.no/~karlsk/dle/index.html
% alternative functions:
%  - GreedLab (sparsify), Thomas Blumensath et al. (Edinburgh)
%  - SparseLab, David Donoho et al. (Stanford)
%  - SPAMS (Mairal):  http://spams-devel.gforge.inria.fr/
%  - OMPBox (Ron Rubinstein): http://www.cs.technion.ac.il/~ronrubin/software.html

mfile = 'sparseapprox';

%% Check if input arguments are given
if (nargin < 3)  % just check number of input arguments 
    t = [mfile,': arguments must be given, see help.'];
    disp(t);
    if nargout >= 1
        varargout{1} = -1;
    end
    if nargout >= 2
        varargout{2} = struct('Error',t);
    end
    return
end

%% defaults, initial values
tstart = tic;
[N,L] = size(X);
K = size(D,2);
norm2X = sqrt(sum(X.*X)); % ||x(i)||_2     1xL
W = zeros(K,L);      % the weights (coefficients)
tnz = ceil(N/2)*ones(1,L); % target number of non-zeros
thrActive = false;    % is set to true if tnz, tre or tae is given
                      % and used for methods: pinv, backslash, linprog and
                      % FOCUSS
doOP = true;          % do Orthogonal Projection when thresholding                 
relLim = 1e-6;
tre = relLim*ones(1,L);   % target relative error: ||r|| <= tre*||x||
nComb = 20;           % used only in javaPS
nIt = 20;             % used only in FOCUSS 
pFOCUSS = 0.5;        % used only in FOCUSS 
lambdaFOCUSS = 0;     % used only in FOCUSS
deltaWlimit = 1e-8;   % used only in FOCUSS
GMPLoopSize = 0;      % used only in GMP
globalReDist = 0;     % may be used with javaORMP 
targetSSE = 0;        % may be used with javaORMP 
verbose = 0;
done = false;
javaClass = 'mpv2.MatchingPursuit';   % the important java class
spams_mex_file = 'mexLasso'; % one of the used SPAMS files
    
%% get the options
nofOptions = nargin-3;
optionNumber = 1;
fieldNumber = 1;
while (optionNumber <= nofOptions)
    if isstruct(varargin{optionNumber})
        sOptions = varargin{optionNumber}; 
        sNames = fieldnames(sOptions);
        opName = sNames{fieldNumber};
        opVal = sOptions.(opName);
        % next option is next field or next (pair of) arguments
        fieldNumber = fieldNumber + 1;  % next field
        if (fieldNumber > numel(sNames)) 
            fieldNumber = 1;
            optionNumber = optionNumber + 1;  % next pair of options
        end
    elseif iscell(varargin{optionNumber})
        sOptions = varargin{optionNumber}; 
        opName = sOptions{fieldNumber};
        opVal = sOptions{fieldNumber+1};
        % next option is next pair in cell or next (pair of) arguments
        fieldNumber = fieldNumber + 2;  % next pair in cell
        if (fieldNumber > numel(sOptions)) 
            fieldNumber = 1;
            optionNumber = optionNumber + 1;  % next pair of options
        end
    else
        opName = varargin{optionNumber};
        opVal = varargin{optionNumber+1};
        optionNumber = optionNumber + 2;  % next pair of options
    end
    % interpret opName and opVal
    if strcmpi(opName,'targetNonZeros') || strcmpi(opName,'tnz')
        if strcmpi(met,'GMP') 
            tnz = opVal;   % GMP will distribute the non-zeros
        else
            if numel(opVal)==1
                tnz = opVal*ones(1,L);
            elseif numel(opVal)==L
                tnz = reshape(opVal,1,L);
            else
                error([mfile,': illegal size of value for option ',opName]);
            end
        end
        thrActive = true;
    end
    if strcmpi(opName,'targetRelativeError') || strcmpi(opName,'tre')
        if numel(opVal)==1
           tre = opVal*ones(1,L);
        elseif numel(opVal)==L
           tre = reshape(opVal,1,L);
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
        thrActive = true;
    end
    if strcmpi(opName,'targetAbsoluteError') || strcmpi(opName,'tae')
        if numel(opVal)==1
           tae = opVal*ones(1,L);
        elseif numel(opVal)==L
           tae = reshape(opVal,1,L);
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
        thrActive = true;
    end
    if ( strcmpi(opName,'nIt') || strcmpi(opName,'nofIt') || ...
         strcmpi(opName,'numberOfIterations') )
        if (isnumeric(opVal) && numel(opVal)==1)
            nIt = max(floor(opVal), 1);
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
    end
    if strcmpi(opName,'p') || strcmpi(opName,'pFOCUSS')
        if (isnumeric(opVal) && numel(opVal)==1)
            pFOCUSS = min(opVal, 1);
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
    end
    if strcmpi(opName,'l') || strcmpi(opName,'lambdaFOCUSS')
        if (isnumeric(opVal) && numel(opVal)==1)
            lambdaFOCUSS = abs(opVal);
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
    end
    if strcmpi(opName,'nComb')
        if (isnumeric(opVal) && numel(opVal)==1)
            nComb = max(floor(opVal), 2);
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
    end
    if strcmpi(opName,'paramSPAMS')
        if (isstruct(opVal))
            paramSPAMS = opVal;
        else
            error([mfile,': option paramSPAMS is not a struct as it should be, see SPAMS help.']);
        end
    end
    if strcmpi(opName,'globalReDist')
        if (isnumeric(opVal) && numel(opVal)==1)
            globalReDist = min(max(floor(opVal), 0), 2);  % 0, 1 or 2
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
    end
    if strcmpi(opName,'doOP') 
        if (islogical(opVal)); doOP = opVal; end;
        if isnumeric(opVal); doOP = (opVal ~= 0); end;
    end
    if strcmpi(opName,'GMPLoopSize')
        if (isnumeric(opVal) && numel(opVal)==1)
            GMPLoopSize = max(floor(opVal), 2);
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
    end
    if strcmpi(opName,'tSSE') || strcmpi(opName,'targetSSE')
        if (isnumeric(opVal) && numel(opVal)==1)
            targetSSE = min(max(opVal, 0), sum(sum(X.*X)));
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
    end
    if strcmpi(opName,'tSNR') || strcmpi(opName,'targetSNR')
        if (isnumeric(opVal) && numel(opVal)==1)
            targetSSE = 10^(-abs(opVal)/10) * sum(sum(X.*X));
        else
            error([mfile,': illegal size of value for option ',opName]);
        end
    end
    if strcmpi(opName,'verbose') || strcmpi(opName,'v')
        if (islogical(opVal) && opVal); verbose = 1; end;
        if isnumeric(opVal); verbose = opVal(1); end;
    end
end

if exist('tae','var')   % if both exist 'tae' overrules 'tre'
    tre = tae./norm2X;
elseif exist('tre','var')  % 'tre' was given a default value
    tae = tre.*norm2X;
else                       % so this case is redundant
    disp(' ??? This is never printed.');
end

%% Display info
if (verbose > 1)  % very verbose
    disp(' ');
    disp([mfile,' with method ',met,' started ',datestr(now)]);
    disp(['Size of X is ',int2str(size(X,1)),'x',int2str(size(X,2)),...
          ', D is ',int2str(size(D,1)),'x',int2str(size(D,2)),...
          ', and W is ',int2str(size(W,1)),'x',int2str(size(W,2))]);
end

%%  Method of Frames
if strcmpi(met,'MOF') || strcmpi(met,'MethodOfFrames') || strcmpi(met,'pinv')
    textMethod = 'Method of Frames with pseudoinverse (pinv).';
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    W = pinv(D)*X;
    if thrActive  % then adjust w by setting more to zero
        W = setSmallWeightsToZero(X,D,W,tnz,tae,doOP);
    end
    done = true;
end

%%  Backslash method just find an exact solution with N non-zeros
if strcmpi(met,'BackSlash') || strcmpi(met,'\')
    textMethod = 'Matlab backslash operator.';
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    W = D\X;
    if thrActive  % then adjust w by setting more to zero
        W = setSmallWeightsToZero(X,D,W,tnz,tae,doOP);
    end
    done = true;
end


%%  Basis Pursuit
if strcmpi(met,'BP') || strcmpi(met,'BasisPursuit') || strcmpi(met,'linprog')
    textMethod = 'Basis Pursuit with Matlab function linprog.';
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    f = ones(2*K,1);
    A = [D,-D];
    LB = zeros(2*K,1);         % lower bound for w
    UB = ones(2*K,1)*inf;      % upper bound for w
    for columnNumber = 1:L
        x = X(:,columnNumber);
        w2 = linprog(f,A,x,A,x,LB,UB);     % minimize 1-norm of w 
        W(:,columnNumber) = w2(1:K)-w2((1+K):(2*K));
    end
    if thrActive  % then adjust w by setting more to zero
        W = setSmallWeightsToZero(X,D,W,tnz,tae,doOP);
    end
    done = true;
end

%%  the original (and possible regularized) FOCUSS algorithm
if strcmpi(met,'FOCUSS')
    W = pinv(D)*X;   % initial values
    if (lambdaFOCUSS > 0)  % Regularized FOCUSS
        textMethod = ['FOCUSS with p=',num2str(pFOCUSS),...
            ', and regularization (lambda = ',....
            num2str(lambdaFOCUSS),')',...
            ' and ',int2str(nIt),' iterations.'];
    else
        textMethod = ['FOCUSS with p=',num2str(pFOCUSS),...
            ' and ',int2str(nIt),' iterations.'];
    end
    if thrActive  
        textMethod = char(textMethod, ...
            ' Thresholding is done in the end.');
    end
    if (nargout >= 2)        % keep track of sparseness
        sparseInW = zeros(5,nIt);
        edges = [0, 0.0001, 0.001, 0.01, 0.1, inf];
        changeInW = zeros(nIt,L);
        numberOfIterations = nIt*ones(1,L);
    end
    if (verbose >= 1); disp(char([mfile,': '],textMethod)); end;
    for columnNumber = 1:L
        w = W(:,columnNumber);
        w0 = w;
        x = X(:,columnNumber);
        for i=1:nIt
            Qdiagonal = abs(w).^(1-pFOCUSS/2);
            F = D.*(ones(N,1)*(Qdiagonal'));  % F^{k+1}  in Engan's PhD
            if (lambdaFOCUSS > 0)  % Regularized FOCUSS
                q = F'*( ((F*F'+lambdaFOCUSS*eye(N)) \ x) );
            else  % original FOCUSS
                q = pinv(F) * x;
            end
            w = Qdiagonal.*q;
            change = norm(w-w0);
            if (nargout >= 2)    % keep track of sparseness and more
                m = max(abs(w));
                I = histc(abs(w)/m, edges);
                sparseInW(:,i) = sparseInW(:,i)+I(1:5);
                changeInW(i,columnNumber) = change;
                if (change < deltaWlimit)
                    sparseInW(:,(i+1):nIt) = sparseInW(:,(i+1):nIt) + I(1:5)*ones(1,nIt-i);
                    changeInW((i+1):nIt,columnNumber) = change;
                    numberOfIterations(columnNumber) = i;
                end
            end
            if (change < deltaWlimit); break; end;
            w0 = w;
        end
        W(:,columnNumber) = w;
    end
    if thrActive  % then adjust w by setting more to zero
        W = setSmallWeightsToZero(X,D,W,tnz,tae,doOP);
    end
    done = true;
end

%%  mexOMP or mexLasso algorithm from SPAMS
if (strcmpi(met,'mexOMP') || strcmpi(met,'mexORMP') || ...
    strcmpi(met,'mexLasso') || strcmpi(met,'LARS') || strcmpi(met,'LASSO'))
    % check matlab version
    t = version('-release');
    if (eval(t(1:4)) < 2009) || strcmpi(t,'2009a')
        t = [mfile,': mexLasso and mexOMP need Matlab version >= 2009b. (?)'];
        disp(t);
        if nargout >= 1
            varargout{1} = -1;
        end
        if nargout >= 2
            varargout{2} = struct('Error',t);
        end
        return
    end
    %
    % The way access to SPAMS is checked on the computeres I may use
    if ~(exist(spams_mex_file,'file') == 3)  % mex-file not available (yet)
        start_spams;   % a m-file that comes with SPAMS and is located on a common path 
    end
    if ~(exist(spams_mex_file,'file') == 3)  % mex-file still not available
        t = [mfile,': can not access mexLasso and mexOMP on this computer.'];
        disp(t);
        if nargout >= 1
            varargout{1} = -1;
        end
        if nargout >= 2
            varargout{2} = struct('Error',t);
        end
        return
    end
    %
    if ~exist('paramSPAMS','var')
        if (strcmpi(met,'mexOMP') || strcmpi(met,'mexORMP'))
            paramSPAMS = struct(...
                'eps',    tae.^2, ...
                'L',      int32(tnz) );
            textMethod = 'mexOMP in SPAMS package (by J. Mairal).';
        else
            paramSPAMS = struct(...
                'mode',   1, ...            
                'lambda', tae.^2, ...
                'L',      int32(tnz) );
            textMethod = 'mexLasso in SPAMS package (by J. Mairal).';
        end
    end
    %
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    %
    if (strcmpi(met,'mexOMP') || strcmpi(met,'mexORMP'))
        W = mexOMP(X, D, paramSPAMS);
    else
        W = mexLasso(X, D, paramSPAMS);
    end
    done = true;
end

%% Algorithms implemented in Java
if strcmpi(met(1:min(numel(met),3)),'aff') 
    disp('Affine approximation is essentiallaly that sum of coefficients should be 1.');
    disp('This is achieved by adding a row of constants to both D and X and ');
    disp('then doing ordinary aparse approximation. Do this outside this function ');
    disp('to have better control. Set a to an approriate value, 5 perhaps.');
    disp('Ex:  D = [a*ones(1,K); D]; X = [a*ones(1,L); X];');
    % met = ['java',met(4:end)];   % use java method
    % D = [5*ones(1,K); D];    
    % X = [5*ones(1,L); X];    
    done = false;
end

if strcmpi(met(1:min(numel(met),4)),'java') 
    if (not(exist(javaClass,'class')) && exist('java_access.m','file'))
        java_access;
    end
    if (not(exist(javaClass,'class')) && exist('javaAccess.m','file'))
        javaAccess;  % an older version of java_access (may work if it exist)
    end
    if not(exist(javaClass,'class'))   
        javaErrorMessage = ['No access to ',javaClass,' in static or dynamic Java path.'];
        disp(javaErrorMessage);
        met = met(5:end);
        disp(['Use method ',met,' instead.']);
    else
        jD = mpv2.SimpleMatrix( D );
        if (L == 1)
            jMP = mpv2.MatchingPursuit(jD);
        else
            jDD = mpv2.SymmetricMatrix(K,K);
            jDD.eqInnerProductMatrix(jD);
            jMP = mpv2.MatchingPursuit(jD,jDD);
        end
    end
end

if ( strcmpi(met,'javaMP') || strcmpi(met,'javaMatchingPursuit') || ...
     strcmpi(met,'javaBMP') || strcmpi(met,'javaBasicMatchingPursuit') )   
    textMethod = 'Basic Matching Pursuit, Java implementation.';
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    % note the 'tre' is not used for BMP
    for j=1:L
        if (tnz(j) > 0)
            W(:,j) = jMP.vsBMP(X(:,j), int32(tnz(j)));
        end
    end
    done = true;
end

if strcmpi(met,'javaOMP') || strcmpi(met,'javaOrthogonalMatchingPursuit')
    textMethod = 'Orthogonal Matching Pursuit, Java implementation.';
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    for j=1:L
        if (tnz(j) > 0) && (tre(j) < 1)
            W(:,j) = jMP.vsOMP(X(:,j), int32(tnz(j)), tre(j));
        end
    end
    done = true;
end

if strcmpi(met,'javaORMP') || strcmpi(met,'javaOrderRecursiveMatchingPursuit')
    %
    % This could be as simple as javaOMP, but since globalReDist was
    % reintroduced it is now quite complicated here.
    if (targetSSE > 0)   
        % This is initialization of tre (and tnz ?) for the special case of
        % global distribution of non-zeros where a target sum og squared
        % errors is given as an input argument.
        % Perhaps tnz also should be set to an appropriate value
        % tnz = 2*ones(1,L); 
        tre = sqrt(targetSSE/L)./norm2X;
        globalReDist = 2;
        textMethod = ['javaORMP with global distribution of non-zeros ',...
            'given target SSE (or SNR).'];
    elseif (globalReDist == 1)
        textMethod = ['javaORMP with global distribution of non-zeros ',...
            'keeping the total number of non-zeros fixed.'];
    elseif (globalReDist == 2)
        textMethod = ['javaORMP with global distribution of non-zeros ',...
            'keeping the total SSE fixed.'];
    else
        textMethod = 'Order Recursive Matching Pursuit, Java implementation.';
    end
    %
    % below is the javaORMP lines
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    for j=1:L
        if (tnz(j) > 0) && (tre(j) < 1)
            W(:,j) = jMP.vsORMP(X(:,j), int32(tnz(j)), tre(j));
        end
    end
    % 
    % below is the globalReDist lines
    % ******* START Global distribution of non-zeros.***** 
    % The structure is:  
    %    1. Initializing:  Sm1 <= S <= Sp1  and  SEm1 >= SE >= SEp1
    %    2. Add atoms until SSE is small enough
    %    3. or remove atoms until SSE is large enough
    %    4. Add one atom as long as one (or more) may be removed and the
    %       SSE is reduced
    if (globalReDist > 0)
        % part 1
        R = X - D * W;    % representation error
        S = sum(W ~= 0);  % selected number of non-zeros for each column
        SE = sum(R.*R);   % squared error for each (when S is selected)
        sumSinit = sum(S);
        SSE = sum(SE);
        SSEinit = SSE;      % store initial SSE
        Sp1 = S + 1;        % selected number of non-zeros plus one
        Sp1(Sp1 > N) = N;
        Sm1 = S - 1;        % selected number of non-zeros minus one
        Sm1(Sm1 < 0) = 0;
        SEp1 = zeros(1,L);  % initializing corresponding squared error
        SEm1 = zeros(1,L);  
        for j=1:L
            x = X(:,j);
            if Sp1(j) == S(j)   % == N
                w = W(:,j);
            else
                w = jMP.vsORMP(x, Sp1(j), relLim);
            end
            r = x-D*w;
            SEp1(j) = r'*r;
            if Sm1(j) == 0
                w = zeros(K,1);
            else
                w = jMP.vsORMP(x, Sm1(j), relLim);
            end
            r = x-D*w;
            SEm1(j) = r'*r;
        end
        SEdec = SE-SEp1;   % the decrease in error by selectiong one more
        SEinc = SEm1-SE;   % the increase in error by selectiong one less
        SEinc(S == 0) = inf;  % not possible to select fewer than zero
        addedS = 0;
        removedS = 0;
        addedSE = 0;
        removedSE = 0;
        [valinc, jinc] = min(SEinc); % min increase in SE by removing one atom
        [valdec, jdec] = max(SEdec); % max reduction in SE by adding one atom
        
        if (targetSSE > 0)
            if (SSEinit > targetSSE)  % part 2
                if (verbose > 2)
                    disp(['(part 2 add atoms, target SSE = ',num2str(targetSSE),...
                        ' and initial SSE = ',num2str(SSEinit),')']);
                end
                while (SSE > targetSSE)
                    j = jdec;    % an atom is added to vector j
                    addedS = addedS+1;
                    removedSE = removedSE + valdec;
                    SSE = SSE - valdec;
                    % shift in  Sm1,S,Sp1  and  SEm1,SE,SEp1
                    [Sm1(j), S(j), Sp1(j)] = assign3(S(j), Sp1(j), min(Sp1(j)+1, N));
                    [SEm1(j), SE(j)] = assign2(SE(j), SEp1(j));  % and SEp1(j)=SEp1(j)
                    if (Sp1(j) > S(j))   % the normal case, find new SEp1(j)
                        w = jMP.vsORMP(X(:,j), Sp1(j), relLim);
                        r = X(:,j) - D*w;
                        SEp1(j) = r'*r;
                    end
                    SEinc(j) = SEdec(j);      % SE cost by removing this again
                    SEdec(j) = SE(j)-SEp1(j); % SE gain by adding one more atom
                    %
                    W(:,j) = jMP.vsORMP(X(:,j), S(j), relLim);
                    [valdec, jdec] = max(SEdec);
                end
                [valinc, jinc] = min(SEinc); 
            elseif ((SSEinit+valinc) < targetSSE)  % part 3
                if (verbose > 2)
                    disp(['(part 3 remove atoms, target SSE = ',num2str(targetSSE),...
                        ' and initial SSE = ',num2str(SSEinit),')']);
                end
                while ((SSE+valinc) < targetSSE)
                    j = jinc;   % an atom is removed from vector j
                    removedS = removedS+1;
                    addedSE = addedSE + valinc;
                    SSE = SSE + valinc;
                    % shift in  Sm1,S,Sp1  and  SEm1,SE,SEp1
                    [Sm1(j), S(j), Sp1(j)] = assign3(max(Sm1(j)-1, 0), Sm1(j), S(j));
                    [SE(j), SEp1(j)] = assign2(SEm1(j), SE(j)); % and SEm1(j)=SEm1(j)
                    if (Sm1(j) > 0)
                        w = jMP.vsORMP(X(:,j), Sm1(j), relLim);
                        r = X(:,j) - D*w;
                    else
                        r = X(:,j);
                    end
                    SEm1(j) = r'*r;
                    %
                    SEdec(j) = SEinc(j);  % SE gain by adding this atom again
                    if (S(j) > 0) % SE cost by removing another atom 
                        W(:,j) = jMP.vsORMP(X(:,j), S(j), relLim);
                        SEinc(j) = SEm1(j)-SE(j);    
                    else
                        W(:,j) = zeros(K,1);
                        SEinc(j) = inf;  % can not select fewer and increase error
                    end
                    [valinc, jinc] = min(SEinc); 
                end
                [valdec, jdec] = max(SEdec);
            else  %
                if (verbose > 2)
                    disp(['(target SSE = ',num2str(targetSSE),...
                        ' is close to initial SSE = ',num2str(SSEinit),')']);
                end
            end
        else  
            targetSSE = SSEinit;
        end 
        %
        % part 4
        while ((valinc < valdec) && (jinc ~= jdec))  
            j = jdec;
            addedS = addedS+1;
            removedSE = removedSE + valdec;
            SSE = SSE - valdec;
            % shift in  Sm1,S,Sp1  and  SEm1,SE,SEp1
            [Sm1(j), S(j), Sp1(j)] = assign3(S(j), Sp1(j), min(Sp1(j)+1, N));
            [SEm1(j), SE(j)] = assign2(SE(j), SEp1(j));  % and SEp1(j)=SEp1(j)
            if (Sp1(j) > S(j))   % the normal case, find new SEp1(j)
                w = jMP.vsORMP(X(:,j), Sp1(j), relLim);
                r = X(:,j) - D*w;
                SEp1(j) = r'*r;
            end
            SEinc(j) = SEdec(j);      % SE cost by removing this again
            SEdec(j) = SE(j)-SEp1(j); % SE gain by adding one more atom
            W(:,j) = jMP.vsORMP(X(:,j), S(j), relLim);
            [valinc, jinc] = min(SEinc);
            %
            while ((SSE+valinc) < targetSSE) 
                j = jinc;
                removedS = removedS+1;
                addedSE = addedSE + valinc;
                SSE = SSE + valinc;
                % shift in  Sm1,S,Sp1  and  SEm1,SE,SEp1
                [Sm1(j), S(j), Sp1(j)] = assign3(max(Sm1(j)-1, 0), Sm1(j), S(j));
                [SE(j), SEp1(j)] = assign2(SEm1(j), SE(j)); % and SEm1(j)=SEm1(j)
                if (Sm1(j) > 0)
                    w = jMP.vsORMP(X(:,j), Sm1(j), relLim);
                    r = X(:,j) - D*w;
                else
                    r = X(:,j);
                end
                SEm1(j) = r'*r;
                %
                SEdec(j) = SEinc(j);  % SE gain by adding this atom again
                if (S(j) > 0) % SE cost by removing another atom 
                    W(:,j) = jMP.vsORMP(X(:,j), S(j), relLim);
                    SEinc(j) = SEm1(j)-SE(j);    
                else
                    W(:,j) = zeros(K,1);
                    SEinc(j) = inf;  % can not select fewer and increase error
                end
                [valinc, jinc] = min(SEinc); 
                if (globalReDist == 1); break; end;
            end
            [valdec, jdec] = max(SEdec);    % next now
        end
        %
        if (verbose > 2)
            disp(['Using globalReDist=',int2str(globalReDist),': ',...
                'non-zeros in W changed as ',int2str(sumSinit),...
                ' + ',int2str(addedS),' - ',int2str(removedS),...
                ' = ',int2str(sum(sum(W ~= 0)))]); 
            disp(['SSE changed as ',num2str(SSEinit),...
                ' + ',num2str(addedSE),' - ',num2str(removedSE),...
                ' = ',num2str(SSE),' = ',num2str(SSEinit+addedSE-removedSE)]); 
            disp(['(target SSE = ',num2str(targetSSE),...
                ' and actual SSE = ',num2str(sum(sum((X - D * W).^2))),')']); 
        end
    end
    %
    % ******* END Global distribution of non-zeros.***** 
    %
    %
    done = true;
end

if strcmpi(met,'javaPS') || strcmpi(met,'javaPartialSearch')
    textMethod = ['Partial Search with ',...
            int2str(nComb),' number of combinations.'];
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    for j=1:L
        if (tnz(j) > 0) && (tre(j) < 1)
            if (tnz(j) < 2)
                W(:,j) = jMP.vsORMP(X(:,j), int32(tnz(j)), tre(j));
            else
                W(:,j) = jMP.vsPS(X(:,j), int32(tnz(j)), tre(j), int32(nComb));
            end
        end
    end
    done = true;
end

%% GMP is Global Matching Pursuit
if strcmpi(met,'GMP')
    if (GMPLoopSize <= 0)
        GMPLoopSize = floor(tnz/N);
    end
    if (GMPLoopSize > 0.9*L)
        GMPLoopSize = floor(0.9*L);
    end
    textMethod = ['Global Matching Pursuit with ',...
            int2str(tnz),' non-zeros, (N=',int2str(N),...
            ', L=',int2str(L),' GMPLoopSize=',int2str(GMPLoopSize),').'];
    if verbose    
        disp(textMethod);    
    end
    Gd = (1./sqrt(sum(D.*D)));        %                G = diag(Gd)
    F = D.*(ones(size(D,1),1)*Gd);    % normalize D,   F = D*G
    %
    nzW = 0;
    R = X;
    while (nzW < tnz)
        if (verbose > 2)
            disp(['GMP: nzW = ',int2str(nzW),' and ||R||^2 = ',...
                num2str(sum(sum((R.*R))))]);
        end
        U = F'*R;    % inner products = G*D'*R
        [um,iK] = max(abs(U));
        [temp,iL] = sort(um,2,'descend'); %#ok<ASGLU>
        for i=1:min(tnz-nzW, GMPLoopSize)
            il = iL(i);
            ik = iK(il);
            W(ik,il) = W(ik,il) + U(ik,il)*Gd(ik);
        end
        nzW = sum(W(:) ~= 0);
        R = X - D*W;
    end
    if (verbose > 1)
        disp(['GMP: nzW = ',int2str(nzW),' and ||R||^2 = ',...
            num2str(sum(sum((R.*R))))]);
    end
    done = true;
end

%% OMP and ORMP are almost equal.
% function is from FrameTools\VSormp.m (ver 12.06.2003 with some few
% modifications). Variables have short names so there is a risk for
% variable mixup when using the algoritm within a larger context.
if strcmpi(met,'OMP') || strcmpi(met,'ORMP')
    if strcmpi(met,'OMP')
        textMethod = 'Orthogonal Matching Pursuit.';
    end
    if strcmpi(met,'ORMP')
        textMethod = 'Order Recursive Matching Pursuit.';
    end
    if (verbose >= 1); disp([mfile,': ',textMethod]); end;
    F = D.*(ones(size(D,1),1)*(1./sqrt(sum(D.*D))));  % normalize D
    FF = F'*F;
    for columnNumber = 1:L
        % **********************  INITIALIZE  **********************
        x = X(:,columnNumber);
        S = tnz(columnNumber);
        r = zeros(S,K);
        w = zeros(K,1);
        T = 1:K;
        e = ones(K,1);
        u = ones(K,1);
        c = x'*F;
        n2x = x'*x;
        n2xLim = n2x*tre(columnNumber);
        % select the first frame vector
        [cm,km] = max(abs(c));
        s = 1;
        J = km;
        T(km) = -1;
        r(1,km) = u(km);
        n2x = n2x-cm*cm;
        % **********************  THE MAIN LOOP  **********************
        while ((s<S) && (n2x>n2xLim))
            for k=1:K
                if (T(k)>=0)
                    r(s,k) = FF(km,k);
                    for n=1:(s-1)
                        r(s,k) = r(s,k)-r(n,km)*r(n,k);
                    end
                    if (u(km)~=0); r(s,k) = r(s,k)/u(km); end;
                    c(k) = c(k)*u(k)-c(km)*r(s,k);
                    if strcmpi(met,'OMP')  % use next line for OMP
                        w(k) = abs(c(k));  % use w here (instead of a new variable d)
                    end
                    e(k) = e(k)-r(s,k)*r(s,k);
                    u(k) = sqrt(abs(e(k)));  % abs kun i matlab!
                    if (u(k)~=0); c(k) = c(k)/u(k); end;
                    if strcmpi(met,'ORMP')  % use next line for ORMP
                        w(k) = abs(c(k));     % use w here (instead of a new variable d)
                    end
                end
            end
            w(km) = 0;   % w(J) = 0;
            % select the next frame vector
            [temp,km] = max(abs(w)); %#ok<ASGLU>
            s = s+1;
            J(s) = km;
            T(km) = -1;
            r(s,km) = u(km);
            cm = c(km);
            n2x = n2x-cm*cm;
        end  % ******** END OF MAIN LOOP **********************************
        
        % ************ BACK-SUBSTITUTION *************
        w = zeros(K,1);
        for k=s:(-1):1
            Jk=J(k);
            for n=s:(-1):(k+1)
                c(Jk) = c(Jk)-c(J(n))*r(k,J(n));
            end
            if (r(k,Jk)~=0); c(Jk) = c(Jk)/r(k,Jk); end;
            w(Jk) = c(Jk);
        end
        %
        W(:,columnNumber) = w;
    end
    W = W .* ((1./sqrt(sum(D.*D)))'*ones(1,L));  % rescale W
    done = true;
end

%% Now we are finished, 'done' should be true
%  but test this before finding (and/or) displaying results
W = full(W);   
% W = sparse(W);     % is a good alternative

%% may display info before returning
if done && ((verbose > 1) || (nargout >= 2))  % need some results
    R = X - D*W;
    varX = var(X(:));     
    varR = var(R(:));     
    if (varR > 0)
        snr = 10*log10(varX/varR);
    else
        snr = inf;
    end
    norm0X = sum(X ~= 0);
    norm1X = sum(abs(X));
    normiX = max(abs(X));  
    norm0R = sum(R ~= 0);
    norm1R = sum(abs(R));
    norm2R = sqrt(sum(R.*R));  
    normiR = max(abs(R));  
    norm0W = sum(W ~= 0);
    norm1W = sum(abs(W));
    norm2W = sqrt(sum(W.*W));  
    normiW = max(abs(W));  
end
if done && (verbose >= 2)  % very verbose
    if (snr < 100)
        disp([mfile,': SNR for the reconstruction is ',...
              num2str(snr,'%7.4f')]);
    elseif (snr < 500)
        disp([mfile,': Almost perfect reconstruction, SNR > 100.']);
    else
        disp([mfile,': Perfect reconstruction, X = D*W.']);
    end
    disp(['Number of non-zeros in W is ',int2str(sum(norm0W)),...
        ' (sparseness factor is ',num2str(sum(norm0W)/(N*L)),')']);
    if exist('numberOfIterations','var');        
        disp(['Average number of iterations for each column : ',...
            num2str(mean(numberOfIterations),'%5.1f')]);
    end
    %
    disp(['X: ',num2str(min(norm0X)),' <= ||x||_0 <= ',...
        num2str(max(norm0X)),'  and mean is ',num2str(mean(norm0X))]);
    disp(['   ',num2str(min(norm1X)),' <= ||x||_1 <= ',...
        num2str(max(norm1X)),'  and mean is ',num2str(mean(norm1X))]);
    disp(['   ',num2str(min(norm2X)),' <= ||x||_2 <= ',...
        num2str(max(norm2X)),'  and mean is ',num2str(mean(norm2X))]);
    disp(['   ',num2str(min(normiX)),' <= ||x||_inf <= ',...
        num2str(max(normiX)),'  and mean is ',num2str(mean(normiX))]);
    disp(['R: ',num2str(min(norm0R)),' <= ||r||_0 <= ',...
        num2str(max(norm0R)),'  and mean is ',num2str(mean(norm0R))]);
    disp(['   ',num2str(min(norm1R)),' <= ||r||_1 <= ',...
        num2str(max(norm1R)),'  and mean is ',num2str(mean(norm1R))]);
    disp(['   ',num2str(min(norm2R)),' <= ||r||_2 <= ',...
        num2str(max(norm2R)),'  and mean is ',num2str(mean(norm2R))]);
    disp(['   ',num2str(min(normiR)),' <= ||r||_inf <= ',...
        num2str(max(normiR)),'  and mean is ',num2str(mean(normiR))]);
    disp(['W: ',num2str(min(norm0W)),' <= ||w||_0 <= ',...
        num2str(max(norm0W)),'  and mean is ',num2str(mean(norm0W))]);
    disp(['   ',num2str(min(norm1W)),' <= ||w||_1 <= ',...
        num2str(max(norm1W)),'  and mean is ',num2str(mean(norm1W))]);
    disp(['   ',num2str(min(norm2W)),' <= ||w||_2 <= ',...
        num2str(max(norm2W)),'  and mean is ',num2str(mean(norm2W))]);
    disp(['   ',num2str(min(normiW)),' <= ||w||_inf <= ',...
        num2str(max(normiW)),'  and mean is ',num2str(mean(normiW))]);
    disp(' ');
    disp([mfile,' with ',met,' done. Used time is ',num2str(toc(tstart))]);
    disp(' ');
end

%% Return Outputs
if done
    if (nargout >= 1)
        varargout{1} = W;
    end
    if (nargout >= 2)
        varargout{2} = struct( 'time', toc(tstart), 'snr', snr, ...
            'textMethod', textMethod, ...
            'norm0X', norm0X, 'norm1X', norm1X, ...
            'norm2X', norm2X, 'normiX', normiX, ...
            'norm0R', norm0R, 'norm1R', norm1R, ...
            'norm2R', norm2R, 'normiR', normiR, ...
            'norm0W', norm0W, 'norm1W', norm1W, ...
            'norm2W', norm2W, 'normiW', normiW );
        % extra output arguments for standard and regularized FOUCUSS 
        if exist('sparseInW','var');        
            varargout{2}.sparseInW = sparseInW;
            varargout{2}.edges = edges;
        end
        if exist('changeInW','var');        
            varargout{2}.changeInW = changeInW;
        end
        if exist('numberOfIterations','var');        
            varargout{2}.numberOfIterations = numberOfIterations;
        end
        if exist('sol','var');        
            varargout{2}.sol = sol;
        end
        if exist('paramSPAMS','var');        
            varargout{2}.paramSPAMS = paramSPAMS;
        end
    end
else  %  ~done      did not find W
    textMethod = ['Unknown method ',met,'. Coefficients W not found.'];
    disp([mfile,': ',textMethod]);
    if (nargout >= 1)
        varargout{1} = [];
    end
    if (nargout >= 2)
        varargout{2} = struct( 'time', toc(tstart), ...
            'textMethod', textMethod );
    end
end
if exist('javaErrorMessage','var');        
    varargout{2}.Error = javaErrorMessage;
end

return

%%
function W = setSmallWeightsToZero(X,D,W,tnz,tae,doOP)
[K,L] = size(W);  % K=size(D,2), L=size(X,2)
for i = 1:L
    [absw,I] = sort(abs(W(:,i)),'descend');
    w = zeros(K,1);
    for s = 1:tnz(i)
        if (absw(s) < 1e-10); break; end;
        if doOP % do orthogonal projection onto the selected columns of D
            Di = D(:,I(1:s));
            w(I(1:s)) = (Di'*Di)\(Di'*X(:,i));
        else  % simple thresholding keeps the largest coefficients unaltered
            w(I(s)) = W(I(s),i);
        end
        r = X(:,i) - D*w;
        if (sqrt(r'*r) < tae(i)); break; end;
    end
    W(:,i) = w;
end
return
        
function [a, b] = assign2(c, d)
a = c;
b = d;
return

function [a, b, c] = assign3(d, e, f)
a = d;
b = e;
c = f;
return


