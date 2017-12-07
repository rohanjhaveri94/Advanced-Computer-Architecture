#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_
#define HISTOGRAM64_BIN_COUNT 64

__global__ void hist_gen(int* d_In,int* p_hist,int N, int total_threads)
{	
//// Kernel to run histogram
	
        __shared__ int sharedMem[4096]; //shared mem
	int i;
	unsigned int g_id = blockIdx.x*blockDim.x + threadIdx.x;

	for(i=0; i<HISTOGRAM64_BIN_COUNT; i++)
		sharedMem[(threadIdx.x * HISTOGRAM64_BIN_COUNT) + i] = 0;

	for(i=g_id; i<N; i+=total_threads){
		sharedMem[(threadIdx.x * HISTOGRAM64_BIN_COUNT) + d_In[i]]++;
	}
	__syncthreads();
	
	for(i=0; i<HISTOGRAM64_BIN_COUNT; i++){
		p_hist[(g_id * HISTOGRAM64_BIN_COUNT) + i]=sharedMem[(threadIdx.x * HISTOGRAM64_BIN_COUNT) + i];
	}
}

__global__ void merge_hist(int* p_hist, int* g_hist, int total_par_hist, int total_threads)
{	
   //// Function to merge the kernel outputs. 

	int i;
	unsigned int g_id = blockIdx.x * blockDim.x + threadIdx.x;
	g_hist[g_id] = 0;
	for(i=g_id; i<total_par_hist*HISTOGRAM64_BIN_COUNT; i+=HISTOGRAM64_BIN_COUNT){
		g_hist[g_id] += p_hist[i];
	}	
	
}
#endif // _FILTER_KERNEL_H_
