
//
// C++/Eigen reimplementatino of features_omp_8x8_1.m
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <iostream>
#include <numeric>
#include <time.h>
#include <assert.h>

#include "mex.h"

#include "omp_eigen.h"

using namespace std;
using namespace Eigen;



const float eps=1e-6;


inline void normalize(MatrixXf& hist, int norm, float threshold)
{
  switch(norm)
  {
    case 0:  break;
    case 1:
      for(int i=0; i<hist.rows(); i++)
        hist.row(i) /= max( hist.row(i).array().abs().sum(), threshold );
      break;
    case 2:
      for(int i=0; i<hist.rows(); i++)
        hist.row(i) /= max( hist.row(i).norm(), threshold );
      break;
    case 3:
      for(int i=0; i<hist.rows(); i++)
        hist.row(i) /= max( pow( hist.row(i).array().abs().pow(3).sum() , 1/3 ), threshold );
      break;
    default:
      cout << "Error: unknown normalization" << endl;
  }
  //hist.rowwise().normalize();
  return;
}

inline void copy_float_to_double(double* dst, float* src, int n)
{
  float* x=src;
  double* y=dst;
  for(int i=0; i<n; i++,x++,y++)
    *y=(double)(*x);
  return;
}

inline void copy_double_to_float(float* dst, double* src, int n)
{
  double* x=src;
  float* y=dst;
  for(int i=0; i<n; i++,x++,y++)
    *y=(float)(*x);
  return;
}

inline void padarray(MatrixXf& im, int im_h, int im_w, int im_channel, int pad_y, int pad_x)
{
  // replace im with im_pad
  assert(pad_y>=0); assert(pad_x>=0);
  int im_h_pad=im_h+pad_y*2;
  int im_w_pad=im_w+pad_y*2;
  MatrixXf im_pad(im_h_pad,im_w_pad*im_channel);

  for(int c=0; c<im_channel; c++)
  {
    im_pad.block(pad_y,pad_x+im_w_pad*c,im_h,im_w)=im.block(0,im_w*c,im_h,im_w);
    // replicate boundary
    im_pad.block(0,0+im_w_pad*c,pad_y,im_w_pad).rowwise() = im_pad.block(pad_y,im_w_pad*c,1,im_w_pad).row(0);
    im_pad.block(im_h_pad-1-pad_y+1,0+im_w_pad*c,pad_y,im_w_pad).rowwise() = im_pad.block(im_h_pad-1-pad_y,im_w_pad*c,1,im_w_pad).row(0);
    im_pad.block(0,0+im_w_pad*c,im_h_pad,pad_x).colwise() = im_pad.block(0,pad_x+im_w_pad*c,im_h_pad,1).col(0);
    im_pad.block(0,im_w_pad-1-pad_x+1+im_w_pad*c,im_h_pad,pad_x).colwise() = im_pad.block(0,im_w_pad-1-pad_x+im_w_pad*c,im_h_pad,1).col(0);
  }
  // replace im with im_pad
  im=im_pad;
  return;
}

inline void im2col(MatrixXf& X, const MatrixXf& im, int block_y, int block_x)
{
  int im_h=im.rows();
  int im_w=im.cols();
  int nf=block_y*block_x;
  int np=(im_h-block_y+1)*(im_w-block_x+1);

  if (X.rows()!=nf || X.cols()!=np)
    X.resize(nf,np);

  for(int dy=0; dy<block_y; dy++)
    for(int dx=0; dx<block_x; dx++)
    {
      int inc=dy+dx*block_y;
      MatrixXf im_block=im.block(dy,dx,im_h-block_y+1,im_w-block_x+1);
      X.row(inc)= Map<MatrixXf>( im_block.data(), 1, np );
    }
  
  return;
}


inline void im2col_multi(MatrixXf& X, const MatrixXf& im, int block_y, int block_x, int im_channel)
{
  int im_h=im.rows();
  int im_w=im.cols();
     assert( (im_w % im_channel)==0 );
  im_w/=im_channel;
  int nf=block_y*block_x;
  int np=(im_h-block_y+1)*(im_w-block_x+1);

  if (X.rows()!=nf*im_channel || X.cols()!=np)
    X.resize(nf*im_channel,np);

  for(int c=0; c<im_channel; c++) {
    //X.block(c*nf,0,nf,np) = im2col( im.block(0,c*im_w,im_w,im_w), block_y, block_x );
    for(int dy=0; dy<block_y; dy++)
      for(int dx=0; dx<block_x; dx++)
      {
        int inc=dy+dx*block_y+c*block_y*block_x;
        for(int y=dy; y<im_h-block_y+1+dy; y++)
          for(int x=dx; x<im_w-block_x+1+dx; x++)
            X(inc,(y-dy)+(x-dx)*(im_h-block_y+1))=im(y,x+c*im_w);
      }
  }
  return;
}

inline void remove_dc(MatrixXf& X)
{
  //X.rowwise() -= X.colwise().sum();
  for(int i=0; i<X.cols(); i++)
  {
    X.col(i).array() -= X.col(i).mean();
  }
}

void smooth_bilinear(MatrixXf& hist, MatrixXf& codes, int hh, int ww, int ndic, int blocksize)
{
  int ny=floor(hh/blocksize);
  int nx=floor(ww/blocksize);

  if (hist.rows()!=ny*nx || hist.cols()!=ndic)
    hist.resize(ny*nx,ndic);

  assert((blocksize % 2)==0);  // will relax later

  MatrixXi xgrid = RowVectorXi::LinSpaced(ww,1,ww).replicate(hh,1);
  MatrixXi ygrid = VectorXi::LinSpaced(hh,1,hh).replicate(1,ww);

  MatrixXf xp = (xgrid.array().cast<float>()-1+0.5)/blocksize-0.5;
  MatrixXf yp = (ygrid.array().cast<float>()-1+0.5)/blocksize-0.5;
  MatrixXi ixp = xp.cast<int>();       //  -0.4 cast to 0 instead of -1
  MatrixXi iyp = yp.cast<int>();
  MatrixXf vx0 = xp-ixp.cast<float>();
  MatrixXf vy0 = yp-iyp.cast<float>();
  MatrixXf vx1 = 1.0-vx0.array();
  MatrixXf vy1 = 1.0-vy0.array();

  int sparsity=codes.cols()/2;
    assert(codes.cols()==sparsity*2);

  hist.setZero();

  for(int s=0; s<sparsity; s++)
  {
    int ystart, yend, xstart, xend;

    ystart=blocksize/2+1-1; yend=ny*blocksize-blocksize/2-1;
    xstart=blocksize/2+1-1; xend=nx*blocksize-blocksize/2-1;

    // discard everthing that's block/2 from boundary
    for(int y=ystart; y<=yend; y++)
      for(int x=xstart; x<=xend; x++)
      {
        int p=y+x*hh;
        hist( iyp(y,x)+ixp(y,x)*ny, (int)codes(p,2*s) ) += vx1(y,x)*vy1(y,x)*codes(p,2*s+1);
        hist( iyp(y,x)+1+ixp(y,x)*ny, (int)codes(p,2*s) ) += vx1(y,x)*vy0(y,x)*codes(p,2*s+1);
        hist( iyp(y,x)+(ixp(y,x)+1)*ny, (int)codes(p,2*s) ) += vx0(y,x)*vy1(y,x)*codes(p,2*s+1);
        hist( iyp(y,x)+1+(ixp(y,x)+1)*ny, (int)codes(p,2*s) ) += vx0(y,x)*vy0(y,x)*codes(p,2*s+1);
      }
  }

  return;
}


void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    if ((nrhs!=3) || (nlhs!=1)) {
      mexErrMsgTxt("Usage : features = features_omp_mex( im, sbin, encoder );");
    }
    // assume the input image is already [0,1] and converted to gray/ab

    const mxArray* im_mat = prhs[0];
    int sbin = (int)mxGetScalar(prhs[1]);
       //assert(sbin==8);

    int patchsize=(int)mxGetScalar(mxGetField(prhs[2],0,"patchsize"));
      assert((patchsize % 2)==1);
    int blocksize=(int)mxGetScalar(mxGetField(prhs[2],0,"blocksize"));
      assert(blocksize==sbin);   //  deal with different blocksize later
    int ndic=(int)mxGetScalar(mxGetField(prhs[2],0,"dicsize"));
    int sparsity=(int)mxGetScalar(mxGetField(prhs[2],0,"sparsity"));
      assert(sparsity>0);
    int norm=(int)mxGetScalar(mxGetField(prhs[2],0,"norm"));    // use numerical value 0/1/2
      assert(norm==0 || norm==1 || norm==2);
    float threshold=(float)mxGetScalar(mxGetField(prhs[2],0,"threshold"));

    const mxArray* dic_mat=mxGetField(prhs[2],0,"dic");

    // check whether input is single precision
    if( !mxIsDouble(im_mat) ) {
      mexPrintf("Error: im must be double precision\n");
      return;
    }
    if( !mxIsDouble(dic_mat) ) {
      mexPrintf("Error: dic must be double precision\n");
      return;
    }

    int im_w, im_h, im_channel;
    const mwSize* im_size = mxGetDimensions(im_mat);
    im_h=im_size[0];
    im_w=im_size[1];
    if (mxGetNumberOfDimensions(im_mat)<3) {
      im_channel=1;
    } else {
      im_channel=im_size[2];
    }
    int nf=patchsize*patchsize*im_channel;  // input feature size at every pixel

      assert(mxGetM(dic_mat)==nf);
      assert(mxGetN(dic_mat)==ndic);

    MatrixXf im(im_h,im_w*im_channel);     // no multidimensional array for Eigen?
    copy_double_to_float( im.data(), (double*)mxGetData(im_mat), im_w*im_h*im_channel );

    padarray(im,im_h, im_w, im_channel, (patchsize-1)/2,(patchsize-1)/2);

    MatrixXf X;
    im2col_multi( X, im, patchsize, patchsize, im_channel );  // sliding
    //remove_dc(X); // not needed, dic are zero mean
 
    MatrixXf dic(nf,ndic);
    copy_double_to_float( dic.data(), (double*)mxGetData(dic_mat), nf*ndic );

    MatrixXf DtX=(dic.transpose()*X);
    MatrixXf G=(dic.transpose()*dic);

    MatrixXf omp_codes(im_w*im_h,sparsity*2);

    omp( DtX, G, sparsity, omp_codes );

    for(int k=0; k<sparsity; k++)
      omp_codes.col(2*k+1) = omp_codes.col(2*k+1).array().abs();
  
    int ny=(int)(im_h/sbin);
    int nx=(int)(im_w/sbin);

    MatrixXf hist(ny*nx,ndic);

    smooth_bilinear(hist, omp_codes, im_h, im_w, ndic, blocksize);
    normalize(hist,norm,threshold);

    MatrixXf f( (ny-2)*(nx-2),ndic );
    for(int y=1; y<ny-1; y++)
      for(int x=1; x<nx-1; x++)
        f.row((y-1)+(x-1)*(ny-2))=hist.row(y+x*ny);

    if (mxGetField(prhs[2],0,"power_trans")!=NULL) {
      mxArray* power_trans=mxGetField(prhs[2],0,"power_trans");
      int npow=max(mxGetM(power_trans),mxGetN(power_trans));
      double* pows=(double*)mxGetData(power_trans);
      MatrixXf f_pow( (ny-2)*(nx-2),ndic*npow );
      for(int ipow=0; ipow<npow; ipow++)
         f_pow.block(0,ipow*ndic,(ny-2)*(nx-2),ndic)=f.array().pow(pows[ipow]);
      f=f_pow;
    }

    if (mxGetField(prhs[2],0,"pad_zero")!=NULL) {
      int pad_zero=(int)mxGetScalar(mxGetField(prhs[2],0,"pad_zero"));
      if (pad_zero>0) {
        f.conservativeResize((ny-2)*(nx-2),f.cols()+pad_zero);
        f.block(0,f.cols()-pad_zero,(ny-2)*(nx-2),pad_zero).setZero();
      }
    }

    mwSize outDims[3];
    outDims[0]=ny-2; outDims[1]=nx-2; outDims[2]=f.cols();

    plhs[0] = mxCreateNumericArray(3,outDims,mxDOUBLE_CLASS,mxREAL);
    copy_float_to_double( (double*)mxGetData(plhs[0]), f.data(), f.rows()*f.cols() );

   return;
}





