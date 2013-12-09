#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <iostream>
#include <numeric>
#include <assert.h>

#include "mex.h"

using namespace std;
using namespace Eigen;

void vec_assign(VectorXf& dst, const VectorXf& src, const VectorXi& index, int n)
{
  for(int j=0; j<n; j++)
    dst(j)=src(index(j));
}

inline int maxabs_c(const float* f, int m)
{
  int pos=-1;
  float fmax=-1e9;
  for(int i=0; i<m; i++) {
   if (abs(f[i])>fmax) {
     fmax=abs(f[i]);
     pos=i;
   }
  }
  return pos;
}


bool DEBUG=0;
void omp(const MatrixXf& DtX, const MatrixXf& G, int T, MatrixXf& gamma)
{
  unsigned int m = DtX.rows();
  unsigned int L = DtX.cols();

  assert( G.rows()==m );
  assert( G.cols()==m );
  assert( T>=0 );    // sparsity
  assert( gamma.cols()==2*T );
  assert( gamma.rows()==L );

  VectorXf alpha(m);
  VectorXi selected_atoms(m);
  VectorXi ind(T);
  VectorXf c=VectorXf::Zero(T);

  unsigned int allocated_cols=T;
  MatrixXf Lchol( allocated_cols,allocated_cols );
  MatrixXf Gsub( m,allocated_cols );

    VectorXf tempvec1(T);
    VectorXf tempvec2(T);

  //#pragma omp parallel for
  for(int signum=0; signum<L; signum++)
  {
    // ignore residual norm stopping criterion 
    float eps2=0;
    float resnorm=1;

    alpha=DtX.col(signum);

    selected_atoms.setZero(); // =VectorXi::Zero(m);
    for(int i=0; i<T; i++)
    {
      MatrixXf::Index pos;

      alpha.array().abs().maxCoeff(&pos);
      ind(i)=pos;
      selected_atoms(pos)=1;

      Gsub.col(i)=G.col(pos);

      // Cholesky update
      if (i==0) {
        Lchol(0,0)=1;
        c(0)=alpha(pos);
      } else {
        // need backsubst
        vec_assign( tempvec1, Gsub.col(i), ind, i );
        MatrixXf Lchol_sub=Lchol.topLeftCorner(i,i);
        tempvec2.head(i) = Lchol_sub.triangularView<Eigen::Lower>().solve(tempvec1.head(i));
        Lchol.row(i).head(i)=tempvec2.head(i);
        float sum = tempvec2.head(i).squaredNorm();
        Lchol(i,i)=sqrt(1-sum);

      // perform orthogonal projection and compute sparse coefficients
      {
        vec_assign( tempvec1, DtX.col(signum), ind, i+1 );
        MatrixXf Lchol_sub=Lchol.topLeftCorner(i+1,i+1);
        c.head(i+1)=(Lchol_sub.transpose().triangularView<Eigen::Upper>().solve( Lchol_sub.triangularView<Eigen::Lower>().solve(tempvec1.head(i+1))) );
      }

      }

     if (i<T-1) {
     alpha.noalias()=DtX.col(signum) - Gsub.topLeftCorner(m,i+1)*c.head(i+1);
     }

    }

    // return a "sparse matrix"
    for(int j=0; j<T; j++) {
      gamma.row(signum)(2*j)=ind(j);
      gamma.row(signum)(2*j+1)=c(j);
    }

  }

  return;
}


