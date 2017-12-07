#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_
#define HISTOGRAM64_BIN_COUNT 64

__global__ void hist_gen(int* d_In,int* p_hist,int N, int tot_threads, int* atomic_timer, clock_t st)
{	
	__shared__ int sharedMem[4096];
	
	clock_t start_atomic = clock();
    
	int i;
	unsigned int g_id = blockIdx.x*blockDim.x + threadIdx.x;

	for(i=0;i<HISTOGRAM64_BIN_COUNT;i++){
		sharedMem[(threadIdx.x * HISTOGRAM64_BIN_COUNT) + i] = 0;}

	for(i=g_id;i<N;i+=tot_threads){
		sharedMem[(threadIdx.x * HISTOGRAM64_BIN_COUNT) + d_In[i]]++;
	}
	__syncthreads();
	
	for(i=0;i<HISTOGRAM64_BIN_COUNT;i++){
		p_hist[(g_id * HISTOGRAM64_BIN_COUNT) + i]=sharedMem[(threadIdx.x * HISTOGRAM64_BIN_COUNT) + i];
	}

	clock_t stop_atomic = clock();
	if (threadIdx.x == 0){
                 printf("\n Block: %d, start time:%d , end time:%d \n", blockIdx.x, start_atomic, stop_atomic );
		atomic_timer[blockIdx.x] = (int) stop_atomic - start_atomic; 
	}
	
}
__global__ void merge_hist(int* p_hist, int* g_hist, int tot_par_hist, int tot_threads){
	
	int i;
	unsigned int g_id = blockIdx.x*blockDim.x + threadIdx.x;
	g_hist[g_id] = 0;
	for(i=g_id;i<tot_par_hist*HISTOGRAM64_BIN_COUNT;i+=HISTOGRAM64_BIN_COUNT){
		g_hist[g_id] += p_hist[i];
	}	
	
}
#endif // _FILTER_KERNEL_H_
