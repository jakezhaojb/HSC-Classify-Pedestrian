
function f=features_hsc_1( im, sbin, encoder )
%
% function f=features_omp( im, sbin, encoder )
%
%

%assert( sbin==8 );
%assert(encoder.blocksize==sbin);
assert(encoder.numblock==1);
assert( mod(encoder.patchsize,2)==1 );

im=im2double(im);
if max(im(:))>5, im=im/255; end

if isfield(encoder,'halfsize') & encoder.halfsize>0,
  assert(mod(sbin,2)==0);
  im=imresize(im,0.5);
  sbin=sbin/2;
  encoder.blocksize=encoder.blocksize/2;
end

switch(encoder.type),
case 'gray'
  im=rgb2gray(im);
case 'ab'
  im=RGB2Lab(im);
  im=im(:,:,2:3);
otherwise
  error('unknown encoder type');
end


f=features_hsc_1_mex(im,sbin,encoder); 

% Dimension reduction with precomputed SVD
if isfield(encoder,'SVD'),
  if ~isempty(encoder.SVD),
    [ny,nx,nf]=size(f);
    f=reshape(f,ny*nx,nf);
    f=f*encoder.SVD;
    f=reshape(f,[ny nx size(f,2)]);    
  end
end

if isfield(encoder,'dic_second') && isfield(encoder,'sparsity_second'),
  % do a second-layer coding
  [ny,nx,nf]=size(f);
  f=reshape(f,ny*nx,nf);
  f=f';
  omp_codes=omp_eigen( encoder.dic_second'*f, encoder.dic_second'*encoder.dic_second, encoder.sparsity_second );
  omp_codes=abs(omp_codes);
  if isfield(encoder,'norm_second') & encoder.norm_second==2,
    omp_codes=omp_codes./repmat(sqrt(sum(omp_codes.^2,1)),size(omp_codes,1),1);
  end
  if isfield(encoder,'add_second'),
    f=[f' reshape(omp_codes',[ny*nx size(omp_codes,1)])];
    f=reshape(f,[ny nx size(f,2)]);
  else
    f=reshape(omp_codes',[ny nx size(omp_codes,1)]);
  end
  if isfield(encoder,'power_trans_second'),
    f=f.^encoder.power_trans_second;
  end
end

f(isnan(f))=0;

