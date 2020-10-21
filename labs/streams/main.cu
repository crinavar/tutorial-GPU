#include <cuda.h>
#include <stdio.h>

const int N = 1 << 20;

__global__ void kernel(float *x, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main(int argc, char **argv){
	// seteo de numero de streams
	if(argc != 2){
		fprintf(stderr, "run as ./prog numstreams\n");
		exit(EXIT_FAILURE);
	}
	const int num_streams = atoi(argv[1]);
	cudaStream_t streams[num_streams];
	float *data[num_streams];
	// creacion de streams y de datos
	for(int i = 0; i < num_streams; i++){
		printf("creando stream %i\n", i);
		cudaStreamCreate(&streams[i]);
		cudaMalloc(&data[i], N * sizeof(float));
	}

	// ejecucion de cada kernel
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("ejecutando con %i streams....", num_streams); fflush(stdout);
	cudaEventRecord(start);
	for (int i = 0; i < num_streams; i++) {
		// launch one worker kernel per stream
		kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

		kernel<<<1, 1>>>(0, 0);
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	printf("ok\n"); fflush(stdout);
	cudaEventSynchronize(stop);
        float milliseconds = 0.0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time GPU: %f\n", milliseconds );	
	cudaDeviceReset();
	return 0;
}
