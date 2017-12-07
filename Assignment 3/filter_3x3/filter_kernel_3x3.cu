
#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_

__global__ void SobelFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
   __shared__ unsigned char sharedMem[BLOCK_HEIGHT * BLOCK_WIDTH];
   float s_SobelMatrix[9];

    s_SobelMatrix[0] = -1;
    s_SobelMatrix[1] = 0;
    s_SobelMatrix[2] = 1;

    s_SobelMatrix[3] = -2;
    s_SobelMatrix[4] = 0;
    s_SobelMatrix[5] = 2;

    s_SobelMatrix[6] = -1;
    s_SobelMatrix[7] = 0;
    s_SobelMatrix[8] = 1;

   // Computer the X and Y global coordinates
   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

   // Get the Global index into the original image
   int index = y * (width) + x;

   // STUDENT:  Check 1
   // Handle the extra thread case where the image width or height 
   // 
   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();

//Making Sure the border threads don't write and return
if((threadIdx.x<FILTER_RADIUS)||(threadIdx.x>=BLOCK_WIDTH-FILTER_RADIUS))
        return;
if((threadIdx.y<FILTER_RADIUS)||(threadIdx.y>=BLOCK_HEIGHT-FILTER_RADIUS))
        return;

// STUDENT: Make sure only the thread ids should write the sum of the neighbors.
    
float sumX = 0, sumY=0;

    sumX = sharedMem[sharedIndex-9]*s_SobelMatrix[0] + sharedMem[sharedIndex - 8]*s_SobelMatrix[1] + sharedMem[sharedIndex - 7]*s_SobelMatrix[2] + sharedMem[sharedIndex - 1]*s_SobelMatrix[3] + sharedMem[sharedIndex + 1]*s_SobelMatrix[5] + sharedMem[sharedIndex + 7]*s_SobelMatrix[6] + sharedMem[sharedIndex + 8]*s_SobelMatrix[7] + sharedMem[sharedIndex + 9]*s_SobelMatrix[8];

sumY = sharedMem[sharedIndex-9]*s_SobelMatrix[0] + sharedMem[sharedIndex - 8]*s_SobelMatrix[3] + sharedMem[sharedIndex - 7]*s_SobelMatrix[6] + sharedMem[sharedIndex - 1]*s_SobelMatrix[1] + sharedMem[sharedIndex + 1]*s_SobelMatrix[7] + sharedMem[sharedIndex + 7]*s_SobelMatrix[2] + sharedMem[sharedIndex + 8]*s_SobelMatrix[5] + sharedMem[sharedIndex + 9]*s_SobelMatrix[8];

 g_DataOut[index] = abs(sumX) + abs(sumY) > EDGE_VALUE_THRESHOLD ? 255 : 0; 

}

__global__ void AverageFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
    __shared__ unsigned char sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

   // Get the Global index into the original image
   int index = y * (width) + x;

   // Handle the extra thread case where the image width or height
   //
   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();

//Making Sure the border threads don't write and return
if((threadIdx.x<FILTER_RADIUS)||(threadIdx.x>=BLOCK_WIDTH-FILTER_RADIUS))
        return;
if((threadIdx.y<FILTER_RADIUS)||(threadIdx.y>=BLOCK_HEIGHT-FILTER_RADIUS))
        return;

  // STUDENT: write code for Average Filter : use Sobel as base code
float sum = 0;

    sum = sharedMem[sharedIndex-9] + sharedMem[sharedIndex - 8] + sharedMem[sharedIndex - 7] + sharedMem[sharedIndex - 1] + sharedMem[sharedIndex + 1] + sharedMem[sharedIndex + 7] + sharedMem[sharedIndex + 8] + sharedMem[sharedIndex + 9] + sharedMem[sharedIndex] ;

 g_DataOut[index] = sum/9;

}



__global__ void HighBoostFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height,  const int HIGH_BOOST_FACTOR )
{
  __shared__ unsigned char sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

  int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
  int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

  // Get the Global index into the original image
  int index = y * (width) + x;
   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();

//Making Sure the border threads don't write and return
if((threadIdx.x<FILTER_RADIUS)||(threadIdx.x>=BLOCK_WIDTH-FILTER_RADIUS))
        return;
if((threadIdx.y<FILTER_RADIUS)||(threadIdx.y>=BLOCK_HEIGHT-FILTER_RADIUS))
        return;

  // STUDENT: write code for High Boost Filter : use Sobel as base code
  // High Boost Factor is 10  
    float sum = 0;

    sum = sharedMem[sharedIndex-9] + sharedMem[sharedIndex - 8] + sharedMem[sharedIndex - 7] + sharedMem[sharedIndex - 1] + sharedMem[sharedIndex + 1] + sharedMem[sharedIndex + 7] + sharedMem[sharedIndex + 8] + sharedMem[sharedIndex + 9] + sharedMem[sharedIndex] ;

 g_DataOut[index] = CLAMP_8bit(sharedMem[sharedIndex] + HIGH_BOOST_FACTOR * (unsigned char)(sharedMem[sharedIndex] - sum/9));

}


#endif // _FILTER_KERNEL_H_


