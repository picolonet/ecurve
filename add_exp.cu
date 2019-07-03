#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

#include <cuda.h>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()


__global__ void reduce0(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s=1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_gm(int *g_idata, int *g_odata) {
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s=1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
  __syncthreads();
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) warpReduce<blockSize>(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void cpu_sum(int* a, int num) {
  int res = 0;
  for (int i = 0; i < num; i ++) {
    res += a[i];
  } 
}

int main(int argc, char* argv[]) {

   int  array_size = 1048576; 
   int* input = (int*) malloc(array_size * sizeof(int));
   int* output= (int*) malloc(10* sizeof(int));

   srand(time(NULL));

   for (int i = 0; i < array_size; i++) {
    input[i] = rand(); 
   }
   int* gpu_input, *gpu_output;

   cudaSetDevice(0);
   cudaMalloc((void **)&gpu_input, sizeof(int)*array_size);
   cudaMalloc((void **)&gpu_output, sizeof(int)*array_size);
   cudaMemcpy(gpu_input, input, sizeof(int)*array_size, cudaMemcpyHostToDevice);

   clock_t start, end;
   start = clock();
   for (int i = 0; i < 100; i++) {
     reduce0<<<8192, 128, 128 * sizeof(int)>>>(gpu_input, gpu_output);
     cudaDeviceSynchronize();
     reduce0<<<64, 128, 128 * sizeof(int)>>>(gpu_output, &(gpu_output[8192]));
     cudaDeviceSynchronize();
     reduce0<<<1, 64, 64 * sizeof(int)>>>(&(gpu_output[8192]), gpu_output);
     cudaDeviceSynchronize();
   }
   end = clock();
   cudaMemcpy(output, gpu_output, sizeof(int)*10, cudaMemcpyDeviceToHost);
   printf("%d", output[0]);
   double time_iter;
   time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
   printf("\n Reduce 0 time = %10.9f ms.\n", time_iter); 

   start = clock();
   for (int i = 0; i < 100; i++) {
     reduce6<128> <<<8192, 128, 128 * sizeof(int)>>>(gpu_input, gpu_output, array_size);
     cudaDeviceSynchronize();
   }
   end = clock();
   cudaMemcpy(output, gpu_output, sizeof(int)*10, cudaMemcpyDeviceToHost);
   printf("%d", output[0]);
   time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n Reduce 6 time = %10.9f ms.\n", time_iter); 

   start = clock();
   for (int i = 0; i < 100; i++) {
     cpu_sum(input, array_size);
   }
   end = clock();
   time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
   printf("\n CPU time = %10.9f ms.\n", time_iter); 
}

