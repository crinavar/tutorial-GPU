#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define BSIZE 1024
#define WARPSIZE 32
#define REAL double

#include "reduction.cuh"

int main(int argc, char **argv){
	if(argc != 4){
		fprintf(stderr, "run as ./prog n modo nt\n");
		exit(EXIT_FAILURE);
	}
	REAL *a, *ad, *out, *outd;
	int n = atoi(argv[1]);
	int m = atoi(argv[2]);
	int nt = atoi(argv[3]);
	a = (REAL*)malloc(sizeof(REAL)*n);
	out = (REAL*)malloc(sizeof(REAL)*1);
	*out = 0.0f;
	for(int i=0; i<n; ++i){
		a[i] = (REAL)rand()/RAND_MAX;
		//a[i] = 1.0f
		//a[i] = (REAL)i;
		//printf("%f ", a[i]);
	}
	printf("\n");

	cudaMalloc(&ad, sizeof(REAL)*n);
	cudaMalloc(&outd, sizeof(REAL)*1);
	cudaMemcpy(ad, a, sizeof(REAL)*n, cudaMemcpyHostToDevice);
		
	REAL t1, t2;
	omp_set_num_threads(nt);
	t1 = omp_get_wtime();
	if(m == 1){
		REAL accum = 0.0;
		#pragma omp parallel for reduction(+:accum)
		for(int i=0; i<n; ++i){
			accum += a[i];
		}
		*out = accum;
		printf("[CPU] result = %f\n", *out);
	}
	else if(m == 2){
		dim3 block(BSIZE, 1, 1);
		dim3 grid((n + BSIZE - 1)/BSIZE, 1, 1);
		printf("grid(%i, %i, %i),  block(%i, %i, %i)\n", 
			grid.x, grid.y, grid.z, 
			block.x, block.y, block.z);
		kernel_reduction<REAL><<<grid, block>>>(ad, n, outd);
		cudaDeviceSynchronize();
		cudaMemcpy(out, outd, sizeof(REAL)*1, cudaMemcpyDeviceToHost);
		printf("[GPU] result = %f\n", *out);
	}
	t2 = omp_get_wtime();
	printf("time: %f secs\n", t2-t1);
}
