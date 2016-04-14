//Source: https://kb.iu.edu/d/bdmg
/**********************    mat_mul.cu    ******************************/
  #include <stdlib.h>
  #include <stdio.h>
  
  #define M  256
  #define P  128
  #define N   64
  #define BLKSIZ 16
  
  __global__ void mat_mul(float *Ad, float *Bd, float *Cd);
  
  int main()
  {
    float  A[M*P], *Ad;
    float  B[P*N], *Bd;
    float  C[M*N], *Cd;
    float  D[M*N];
    dim3   blockDim(BLKSIZ,BLKSIZ);
    dim3   gridDim(M/BLKSIZ,N/BLKSIZ);
    int    i;
    int    j,k;
  
  /* Fill A and B with random numbers */
    for(i=0;i<M*P;i++)
      A[i]= rand()/(float)RAND_MAX;
    for(i=0;i<P*N;i++)
      B[i]= rand()/(float)RAND_MAX;
  
  /* First, compute D=AB on the host CPU. */
    for(i=0;i<M;i++) {
      for(j=0;j<N;j++) {
  	D[i*N+j]=0.0;
  	for(k=0;k<P;k++) {
  	  D[i*N+j] += A[i*P+k]*B[k*N+j];
  	}
      }
    }
  
  /* Now compute C=AB on the GPU, using a CUDA kernel.
  * First, allocate device memory on the GPU for the matrices */
    cudaMalloc(&Ad,(size_t)(M*P*sizeof(float)));
    cudaMalloc(&Bd,(size_t)(P*N*sizeof(float)));
    cudaMalloc(&Cd,(size_t)(M*N*sizeof(float)));
  
  /* Copy A and B from host memory to device memory */
    cudaMemcpy(Ad,A,M*P*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(Bd,B,P*N*sizeof(float),cudaMemcpyHostToDevice);
  
  /* Call the CUDA kernel to compute Cd=Ad*Bd. */
    mat_mul<<<gridDim,blockDim>>>(Ad,Bd,Cd);
  
  /* Copy Cd from device memory to C in host memory */
    cudaMemcpy(C,Cd,M*N*sizeof(float),cudaMemcpyDeviceToHost);
  
  /* Then free the allocated arrays in device memory. */
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
  
  /* Finally, print out a few of the matrix elements of A, B,
  * C and D. */
    printf("                                     GPU         CPU \n");
    printf("  i    j     A(i,j)    B(i,j)      C(i,j)      D(i,j)\n");
    for(i=0;i<10;i++) {
      for(j=25;j<28;j++) {
  	printf("%4d  %4d  %9.6f %9.6f %11.6f %11.6f\n",
  		    i,j,A[i*P+j],B[i*N+j],C[i*N+j],D[i*N+j]);
      }
    }
  }
  
  __global__ void mat_mul(float *Ad, float *Bd, float *Cd) {
    int    m = blockIdx.x;
    int    n = blockIdx.y;
    int    i = threadIdx.x;
    int    j = threadIdx.y;
    int    k,p;
    float  c = 0.0;
  
    __shared__  float As[BLKSIZ][BLKSIZ];
    __shared__  float Bs[BLKSIZ][BLKSIZ];
  
    for(p=0;p<P/BLKSIZ;p++) {
      As[i][j] = Ad[(m*BLKSIZ+i)*P+(p*BLKSIZ+j)];
      Bs[i][j] = Bd[(p*BLKSIZ+i)*N+(n*BLKSIZ+j)];
      __syncthreads();
      for(k=0; k<BLKSIZ; k++) {
  	c += As[i][k] * Bs[k][j];
      }
    }
    Cd[(m*BLKSIZ+i)*N+(n*BLKSIZ+j)] = c;
  }
  /**********************************************************************/
