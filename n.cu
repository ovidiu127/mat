#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <cuda_fp16.h>

#define ERR 1e-3
#define TILE 16

void init(float *a, int n){
  srand(time(0));
  for(int i=0;i<n;++i){
    a[i]=(double)rand()/RAND_MAX;
  }
}

__global__ void matrix_multiply_kernel(half *A, half *B, float *C, int N, int K, int M)
{
    // Define shared memory tiles
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    // Define register variables
    float4 a, b, c;
    
    // Define block and thread indexes
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Define the row and column indices of the output element
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    // Initialize the output element to zero
    c.x = 0;
    c.y = 0;
    c.z = 0;
    c.w = 0;
    
    // Loop over all tiles of A and B
    for (int i = 0; i < K; i += 16)
    {
        // Load tiles of A and B into shared memory
        As[ty][tx] = A[row * K + i + tx];
        Bs[ty][tx] = B[(i + ty) * M + col];
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();
        
        // Compute a partial dot product using tensor cores
        a = *((float4*)&As[ty][0]);
        b = *((float4*)&Bs[0][tx]);
        c.x += __fma_rn(a.x, b.x, c.x);
        c.y += __fma_rn(a.x, b.y, c.y);
        c.z += __fma_rn(a.y, b.x, c.z);
        c.w += __fma_rn(a.y, b.y, c.w);
        
        // Synchronize to make sure the partial dot product is done
        __syncthreads();
    }
    
    // Store the output element
    C[row * M + col] = c.x + c.y + c.z + c.w;
}

void matrix_multiply(float *A, float *B, float *C, int N, int K, int M)
{
    // Define device matrices
    half *d_A, *d_B;
    float *d_C;
    
    // Allocate device memory
    cudaMalloc(&d_A, N * K * sizeof(half));
    cudaMalloc(&d_B, K * M * sizeof(half));
    cudaMalloc(&d_C, N * M * sizeof(float));
    
    // Copy input matrices to device memory
    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + 15) / 16, (N + 15) / 16);
    
    // Call the matrix multiplication kernel
    matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, K, M);
    
    // Copy the result from device to host memory
    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
  const int m=2000,n=2000,p=2000;

  float *a,*b,*c;
  a=(float*)malloc(m*n*sizeof(float));
  b=(float*)malloc(n*p*sizeof(float));
  c=(float*)malloc(m*p*sizeof(float));

  init(a,m*n);
  init(b,n*p);

  clock_t beg=clock(),end;
  matrix_multiply(a,b,c,m,n,p);
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