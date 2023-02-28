#include <stdio.h>
#include <assert.h>
#include <time.h>

#define ERR 1e-3
#define TILE 32

void init(float *a, int n){
  srand(time(0));
  for(int i=0;i<n;++i){
    a[i]=((rand()%2==1)?-1:1)*(double)rand()/RAND_MAX;
  }
}

__global__ void ker0(float *a, float *b, float *c, int m, int n, int p){
  int row=blockIdx.y * blockDim.y + threadIdx.y;
  int col=blockIdx.x * blockDim.x + threadIdx.x;

  if(row < m && col < p){
    float tmp=0;
    for(int i=0;i<n;++i){
      tmp += a[row * n + i] * b[i * p + col];
    }
    c[row * p + col] = tmp;
  }
}

__global__ void ker(float *a, float *b, float *c, int m, int n, int p){
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

void mmul(float *a, float *b, float *c, int m, int n, int p){
  float *d_a,*d_b,*d_c;

  cudaMalloc(&d_a,m*n*sizeof(float));
  cudaMalloc(&d_b,n*p*sizeof(float));
  cudaMalloc(&d_c,m*p*sizeof(float));

  cudaMemcpy(d_a,a,m*n*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b,n*p*sizeof(float),cudaMemcpyHostToDevice);

  clock_t beg=clock(),end;

  dim3 blockDim={TILE,TILE}, gridDim={(unsigned)(p+TILE-1)/TILE, (unsigned)(m+TILE-1)/TILE};

  ker<<<gridDim,blockDim>>>(d_a,d_b,d_c,m,n,p);
  cudaDeviceSynchronize();

  end=clock();
  printf("GPU util: %f ms\n",(float)(end-beg)/((float)CLOCKS_PER_SEC/1000));

  cudaMemcpy(c,d_c, m*p*sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void check(float *a, float *b, float *c, int m, int n, int p){
  for(int x=0;x<m;++x){
    for(int y=0;y<p;++y){
      float tmp=0;
      for(int z=0;z<n;++z){
        tmp+=a[x * n + z] * b[z * p + y];
      }
      if(!(abs(c[x * p + y] - tmp) < ERR)){
        printf("Err: %f %f\n",tmp,c[x*p+y]);
      }
      assert(abs(c[x * p + y] - tmp) < ERR);
    }
  }
}

int main(){
  const int m=2000,n=3000,p=2000;

  float *a,*b,*c;
  a=(float*)malloc(m*n*sizeof(float));
  b=(float*)malloc(n*p*sizeof(float));
  c=(float*)malloc(m*p*sizeof(float));

  init(a,m*n);
  init(b,n*p);

  clock_t beg=clock(),end;
  mmul(a,b,c,m,n,p);
  end=clock();
  printf("GPU total: %f ms\n",(float)(end-beg)/((float)CLOCKS_PER_SEC/1000));


  beg=clock();
  check(a,b,c,m,n,p);
  end=clock();
  printf("CPU total: %f ms\n",(float)(end-beg)/((float)CLOCKS_PER_SEC/1000));

  free(a);
  free(b);
  free(c);

  return 0;
}