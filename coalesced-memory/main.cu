#include <cuda.h>
#include <cstdio>
#include <omp.h>

#define BSIZE 512
#define OFFSET 15

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void coalesced_mem(int n, float *A, int *B, float *C){
    // id thread (global)
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int newLocalTid = (threadIdx.x/32)*32 + (31-(threadIdx.x % 32));
    int newTid = blockDim.x*blockIdx.x + newLocalTid;
    // CASO 1: read A, write C, aligned y secuencial
    //C[tid] = A[tid];

    // CASE 2: read A con offset, write C, secuencial y no aligned
    //C[(tid+OFFSET) % n] = A[(tid+OFFSET) % n];

    // CASE 3: read A, write C -> aligned y warp no secuencial
    C[newTid] = A[newTid];

    // CASE 4: read A through B, write C -> aligned global no secuencial
    //C[tid] = A[B[tid]];

}

int main(int argc, char **argv){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1) variables
    if(argc != 2){
        fprintf(stderr, "error ejecutar como ./prog n\n");
        exit(EXIT_FAILURE);
    }
    float *Ah, *Ch, *Ad, *Cd;
    int *Bh, *Bd;
    int n = atoi(argv[1]);
    Ah = new float[n];
    Bh = new int[n];
    Ch = new float[n];


    // 2) inicializacion de datos
    printf("init datos..."); fflush(stdout);
    for(int i=0; i<n; ++i){
        Ah[i] = (float)i;
        Bh[i] = rand() % n;
        Ch[i] = 0.0f;
    }
    printf("done\n"); fflush(stdout);


    // 3) copiar a GPU
    printf("cudaMalloc y cudaMemCpy..."); fflush(stdout);
    cudaMalloc(&Ad, sizeof(float)*n);
    cudaMalloc(&Bd, sizeof(int)*n);
    cudaMalloc(&Cd, sizeof(float)*n);

    cudaMemcpy(Ad, Ah, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, Bh, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, Ch, sizeof(float)*n, cudaMemcpyHostToDevice);
    printf("done\n"); fflush(stdout);


    // 4) ejecutar kernel coalesced_mem
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n + BSIZE - 1)/BSIZE, 1, 1);
    printf("Ejecutando Kernel..."); fflush(stdout);
    cudaEventRecord(start);
    coalesced_mem<<<grid, block>>>(n, Ad, Bd, Cd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    printf("done\n"); fflush(stdout);


    double t1,t2;
    printf("Calculo CPU..."); fflush(stdout);
    t1 = omp_get_wtime();
    #pragma omp parallel for num_threads(12)
    for(int i=0; i<n; ++i){
        Ch[i] = Ah[Bh[i]];
    }
    t2 = omp_get_wtime();
    printf("done: %f ms\n", (t2-t1)*1000.0);


    // 5) tiempos
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Ejecucion kernel --> %f ms\n", milliseconds);
}

