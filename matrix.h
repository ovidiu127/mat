#ifndef __MATRIX
#define __MATRIX

#include <assert.h>
#include <time.h>
#include <stdlib.h>

#define ERR 1e-3
#define TILE 32

template<class T>
class matrix{
private:
  

public:
  T* d=NULL;

  int x=-1,y=-1;

  matrix(){
    x=-1;
    y=-1;
  }

  matrix(int _x, int _y){

    if(d != NULL){
      cudaFree(d);
    }

    x=_x;
    y=_y;

    cudaMallocManaged(&d,x*y*sizeof(T));
  }

  T at(int i){
    return d[i];
  }

  void rinit(int seed=time(0)){
    srand(seed);
    int lim=x*y;
    for(int i=0;i<lim;++i){
      d[i]=(float)rand()/RAND_MAX;
    }
  }
};

__global__ void mmul(float *a, float *b, float *c, int m, int n, int p){
  int tx=threadIdx.x, ty=threadIdx.y;
  int bx=blockIdx.x, by=blockIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  __shared__ float A[TILE][TILE];
  __shared__ float B[TILE][TILE];

  float tmp=0;
  int lim=(n+TILE-1)/TILE;

  for(int i=0;i<lim;++i){
    if(row < m && i * TILE + tx < n){
      A[ty][tx]=a[row * n + i * TILE + tx];
    }
    else{
      A[ty][tx]=0;
    }
    if(col < p && i * TILE + ty < n){
      B[ty][tx]=b[(i * TILE + ty) * p + col];
    }
    else{
      B[ty][tx]=0;
    }

    __syncthreads();

    for(int j=0;j<TILE;++j){
      tmp += A[ty][j] * B[j][tx];
    }

    __syncthreads();
  }
  
  if(row < m && col < p){
    c[row * p + col] = tmp;
  }
}

void dot(const matrix<float> &a, const matrix<float> &b, matrix<float> &c){
  assert(a.y==b.x);
  if(c.x!=a.x || c.y!=b.y){
    c=matrix<float>(a.x,b.y);
  }

  dim3 blockDim={TILE,TILE}, gridDim={(unsigned)(b.y+TILE-1)/TILE, (unsigned)(a.x+TILE-1)/TILE};

  mmul<<<gridDim,blockDim>>>(a.d,b.d,c.d,a.x,a.y,b.y);
  cudaDeviceSynchronize();
}

void check(const matrix<float> &a, const matrix<float> &b, matrix<float> &c){
  for(int x=0;x<a.x;++x){
    for(int y=0;y<b.y;++y){
      float tmp=0;
      for(int z=0;z<a.y;++z){
        tmp+=a.d[x*a.y+z]*b.d[z*b.y+y];
      }
      assert(abs(tmp-c.d[x*c.y+y])<ERR);
    }
  }
}

#endif