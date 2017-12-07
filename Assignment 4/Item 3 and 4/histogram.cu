 #include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <sys/io.h>
#include <cutil_inline.h> 
#include "histogram_kernel.cu"
#define HISTOGRAM64_BIN_COUNT 64
int N;
int ThreadsPerBlock;
int NumBlocks;
int *d_In;
int *p_hist;
int *g_hist;
int *atomic_timer;


void ParseArguments(int, char**);
__global__ void hist_gen(int* d_In,int* p_hist,int N, int tot_threads);
__global__ void merge_hist(int* p_hist, int* g_hist, int tot_par_hist, int tot_threads);
int main(int argc, char** argv)
{
	unsigned int timer_m = 0, timer_p = 0, timer_cpu = 0;
	cutilCheckError(cutCreateTimer(&timer_m));
	cutilCheckError(cutCreateTimer(&timer_p));
	cutilCheckError(cutCreateTimer(&timer_cpu));
    ParseArguments(argc, argv);
	int hist_cpu[HISTOGRAM64_BIN_COUNT],i;
	int a[N],final_hist[HISTOGRAM64_BIN_COUNT],times[NumBlocks];
	int tot_threads = ThreadsPerBlock*NumBlocks;
	int tot_par_hist;
	if(N>tot_threads)
		tot_par_hist = tot_threads;
	else
		tot_par_hist = N;
	
	srand(1);	// set rand() seed to 1 for repeatability 

	for(i=0;i<N;i++) {	// load array with digits
		  a[i] = rand() % HISTOGRAM64_BIN_COUNT;  // Specify the number to be 0-63
	}
	
	
	cutilCheckError(cutStartTimer(timer_cpu));
	for(i = 0; i < HISTOGRAM64_BIN_COUNT; i++){
		hist_cpu[i] = 0;
	}
	for(i = 0; i < N; i++){
		hist_cpu[a[i]]++;
	}
	cutilCheckError(cutStopTimer(timer_cpu));
	printf("CPU Histogram:\n");
        printf("\n....................................\n");
	for(i = 0; i < HISTOGRAM64_BIN_COUNT; i++){
		printf("%d  ",hist_cpu[i]);
	}
	printf("\n");
	
	cutilSafeCall( cudaMalloc( (void **)&d_In, N*sizeof(int)) );
	//cutilSafeCall( cudaMalloc( (void **)&p_hist, tot_par_hist*HISTOGRAM64_BIN_COUNT*sizeof(int)) );
	cutilSafeCall( cudaMalloc( (void **)&g_hist, HISTOGRAM64_BIN_COUNT*sizeof(int)) );
	cutilSafeCall( cudaMalloc( (void **)&atomic_timer, NumBlocks*sizeof(int)) );

	cutilCheckError(cutStartTimer(timer_m));
	cudaMemcpy(d_In, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cutilCheckError(cutStopTimer(timer_m));
	cutilCheckError(cutStartTimer(timer_p));	

	hist_gen<<< NumBlocks, ThreadsPerBlock >>>(d_In, g_hist, N, tot_threads,atomic_timer);	
	cutilCheckError(cutStopTimer(timer_p));
	cutilCheckError(cutStartTimer(timer_m));
	cudaMemcpy(final_hist, g_hist, HISTOGRAM64_BIN_COUNT*sizeof(int), cudaMemcpyDeviceToHost);
	cutilCheckError(cutStopTimer(timer_m));

	cudaMemcpy(times, atomic_timer, NumBlocks*sizeof(int), cudaMemcpyDeviceToHost);

	printf("GPU Histogram: Total Thread=%d\n",tot_threads);
        printf("\n...................................................\n");
	for(i = 0; i < HISTOGRAM64_BIN_COUNT; i++){
		printf("%d  ",final_hist[i]);
	}
        printf(".................................................\n");
	printf("\nBlock times\n");
	for(i = 0; i < NumBlocks; i++){
		printf("%f  ",(double)times[i]/(double)CLOCKS_PER_SEC);
	}
	printf("\n");
	printf("Memory Transfer time: %f (ms) \n", cutGetTimerValue(timer_m));
	printf("Processing time: %f (ms) \n", cutGetTimerValue(timer_p));
	printf("Total GPU implementation time: %f (ms) \n", cutGetTimerValue(timer_p)+cutGetTimerValue(timer_m));
	printf("CPU implementation time: %f (ms) \n", cutGetTimerValue(timer_cpu));
	
	return 0;
}

void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--length") == 0 || strcmp(argv[i], "-length") == 0) {
            N = atoi(argv[i+1]);
	    i = i + 1;
        }
        if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-threads") == 0) {
            ThreadsPerBlock = atoi(argv[i+1]);
	    i = i + 1;
        }
        if (strcmp(argv[i], "--blocks") == 0 || strcmp(argv[i], "-blocks") == 0) {
            NumBlocks = atoi(argv[i+1]);
	    i = i + 1;
         
        }
    }
}
