#ifndef REDUCTION_H
#define REDUCTION_H
/* warp reduction with shfl function */
template < typename T >
__inline__ __device__ T warp_reduce(T val){
	for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1)
		val += __shfl_down_sync(0xFFFF, val, offset, WARPSIZE);
	return val;
}

/* block reduction with warp reduction */
template < typename T >
__inline__ __device__ T block_reduce(T val){
	static __shared__ T shared[WARPSIZE];
	int tid = threadIdx.x; 
	int lane = tid & (WARPSIZE-1);
	int wid = tid/WARPSIZE;
	val = warp_reduce<T>(val);
	if(lane == 0)
		shared[wid] = val;

	__syncthreads();
	val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : 0;
	if(wid == 0){
		val = warp_reduce<T>(val);
	}
	return val;
}

/* magnetization reduction using block reduction */
template <typename T>
__global__ void kernel_reduction(T *a, int N, T *out){
	// offsets
	int tid = blockIdx.x * blockDim.x + threadIdx.x; 
	if(tid < N){
		REAL sum = a[tid];
		sum = block_reduce<T>(sum); 
		if(threadIdx.x == 0){
			atomicAdd(out, sum);
		}
	}
}
#endif
