#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_
#define HISTOGRAM64_BIN_COUNT 64

__global__ void hist_gen(int* d_In, int* g_hist, int N, int tot_threads, int* atomic_timer)
{	
	int i;
 
	unsigned int g_id = blockIdx.x*blockDim.x + threadIdx.x;
     
    //the timer for to calculate atomic 
        clock_t start_atomic = clock();
	for(i=g_id;i<N;i+=tot_threads){
		atomicAdd(&g_hist[d_In[i]], 1);
	}
    // stop
	clock_t stop_atomic = clock();
	__syncthreads();
		
	if (threadIdx.x == 0){
		atomic_timer[blockIdx.x] = (int)(stop_atomic - start_atomic);}
}

#endif // _FILTER_KERNEL_H_
