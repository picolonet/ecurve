#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <cassert>
#include "cgbn/cgbn.h"
#include "utility/support.h"

#include "common.cu"

// #define TPI 32
#define TPI  32 

#define BITS 1024 

#define TPB 128    // the number of threads per block to launch (must be divisible by 32

typedef struct {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> y;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> result1;
  cgbn_mem_t<BITS> result2;
} my_instance_t;

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env1024_t;

const uint64_t MNT4_INV = 0xf2044cfbe45e7fff;
const uint64_t MNT6_INV = 0xc90776e23fffffff;

__global__ void cg_mul_const_kernel(my_instance_t *problem_instances, uint32_t instance_count, uint32_t f) {
  context_t         bn_context;                                 // create a CGBN context
  env1024_t         bn1024_env(bn_context);                     // construct a bn environment for 1024 bit math
  env1024_t::cgbn_t a, b, m, r;              

  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid

  cgbn_load(bn1024_env, a, &(problem_instances[my_instance]).x);

  cgbn_mul_ui32(bn1024_env, r, a, f);
  cgbn_store(bn1024_env, &(problem_instances[my_instance].result1), r);
}

__global__ void cg_add_kernel(my_instance_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context;                                 // create a CGBN context
  env1024_t         bn1024_env(bn_context);                     // construct a bn environment for 1024 bit math
  env1024_t::cgbn_t a, b, m, r;              

  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid

  cgbn_load(bn1024_env, a, &(problem_instances[my_instance]).x);
  cgbn_load(bn1024_env, b, &(problem_instances[my_instance]).y);
  cgbn_load(bn1024_env, m, &(problem_instances[my_instance]).m);

  cgbn_add(bn1024_env, r, a, b);

  cgbn_store(bn1024_env, &(problem_instances[my_instance].result1), r);
}

__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ static int32_t fast_propagate_add(const uint32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    g=__ballot_sync(sync, carry==1);
    p=__ballot_sync(sync, x==0xFFFFFFFF);
 
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);
    
    x=x+(c!=0);
     
    return sum>>32;   // -(p==0xFFFFFFFF);
  }

__device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

  __device__ __forceinline__ static int32_t resolve_add(const int32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    c=__shfl_up_sync(sync, carry, 1);
    c=(warp_thread==0) ? 0 : c;
    x=add_cc(x, c);
    c=addc(0, 0);

    g=__ballot_sync(sync, c==1);
    p=__ballot_sync(sync, x==0xFFFFFFFF);

    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);
    x=x+(c!=0);
  
    c=carry+(sum>>32);
    return __shfl_sync(sync, c, 31);
  }

__global__ void my_mul_const_kernel(my_instance_t *problem_instances, uint32_t instance_count, uint32_t f) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid

  uint32_t lane = threadIdx.x % TPI;
  uint32_t prd, carry = 0;

  cgbn_mem_t<BITS>& a = problem_instances[my_instance].x;
  cgbn_mem_t<BITS>& b = problem_instances[my_instance].y;
  cgbn_mem_t<BITS>& r = problem_instances[my_instance].result2;

  prd = madlo_cc(a._limbs[lane], f, carry);
  carry=madhic(a._limbs[lane], f, 0);
  carry=resolve_add(carry, prd);
  r._limbs[lane] = prd;
}

__global__ void my_add_kernel(my_instance_t *problem_instances, uint32_t instance_count) {
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
 

  uint32_t lane = threadIdx.x % TPI;
  uint32_t sum, carry;
  cgbn_mem_t<BITS>& a = problem_instances[my_instance].x;
  cgbn_mem_t<BITS>& b = problem_instances[my_instance].y;
  cgbn_mem_t<BITS>& r = problem_instances[my_instance].result2;

  sum = add_cc(a._limbs[lane], b._limbs[lane]);
  carry = addc_cc(0, 0);
  fast_propagate_add(carry, sum);

  r._limbs[lane] = sum;
}

std::vector<uint8_t*>* compute_mul_const_cuda(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t* input_m_base, int num_bytes, FILE* debug_file) {

  int num_elements = a.size();

  my_instance_t *gpuInstances;
  my_instance_t* instance_array = (my_instance_t*) malloc(sizeof(my_instance_t) * num_elements);
  cgbn_error_report_t *report;

  // create a cgbn_error_report for CGBN to report back errors
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  for (int i = 0; i < num_elements; i ++) {
    std::memcpy((void*)instance_array[i].x._limbs, (const void*) a[i], num_bytes);
    std::memcpy((void*)instance_array[i].y._limbs, (const void*) b[i], num_bytes);
    std::memcpy((void*)instance_array[i].m._limbs, (const void*) input_m_base, num_bytes);
  }

  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(my_instance_t)*num_elements));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(my_instance_t)*num_elements, cudaMemcpyHostToDevice));
  
  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;
  int tpi = TPI;
  // printf("\n Threads per instance = %d", tpi);
  // printf("\n Instances per block = %d", IPB);

  uint32_t num_blocks = (num_elements+IPB-1)/IPB;
  // printf("\n Number of blocks = %d", num_blocks);

  cg_mul_const_kernel<<<8192, TPB>>>(gpuInstances, num_elements, 13);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  my_mul_const_kernel<<<8192, TPB>>>(gpuInstances, num_elements, 12);
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(my_instance_t)*num_elements, cudaMemcpyDeviceToHost));

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
     if (!std::memcmp(instance_array[i].result1._limbs, instance_array[i].result2._limbs, num_bytes)) {
        printf("\n DO NOT MATCH: %d\n", i);
        fprint_uint8_array(debug_file, (uint8_t*)instance_array[i].result1._limbs, num_bytes); 
        fprint_uint8_array(debug_file, (uint8_t*)instance_array[i].result2._limbs, num_bytes); 
     } else {
        printf("\n MATCH: %d\n", i);
     }
  }

  free(instance_array);
  cudaFree(gpuInstances);
  return res_vector;
}

std::vector<uint8_t*>* compute_add_cuda(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t* input_m_base, int num_bytes, FILE* debug_file) {

  int num_elements = a.size();

  my_instance_t *gpuInstances;
  my_instance_t* instance_array = (my_instance_t*) malloc(sizeof(my_instance_t) * num_elements);
  cgbn_error_report_t *report;

  // create a cgbn_error_report for CGBN to report back errors
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  for (int i = 0; i < num_elements; i ++) {
    std::memcpy((void*)instance_array[i].x._limbs, (const void*) a[i], num_bytes);
    std::memcpy((void*)instance_array[i].y._limbs, (const void*) b[i], num_bytes);
    std::memcpy((void*)instance_array[i].m._limbs, (const void*) input_m_base, num_bytes);
  }

  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(my_instance_t)*num_elements));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(my_instance_t)*num_elements, cudaMemcpyHostToDevice));
  
  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;
  int tpi = TPI;
  // printf("\n Threads per instance = %d", tpi);
  // printf("\n Instances per block = %d", IPB);

  uint32_t num_blocks = (num_elements+IPB-1)/IPB;
  // printf("\n Number of blocks = %d", num_blocks);

  cg_add_kernel<<<8192, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  my_add_kernel<<<8192, TPB>>>(gpuInstances, num_elements);
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(my_instance_t)*num_elements, cudaMemcpyDeviceToHost));

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
     if (!std::memcmp(instance_array[i].result1._limbs, instance_array[i].result2._limbs, num_bytes)) {
        printf("\n DO NOT MATCH: %d\n", i);
        fprint_uint8_array(debug_file, (uint8_t*)instance_array[i].result1._limbs, num_bytes); 
        fprint_uint8_array(debug_file, (uint8_t*)instance_array[i].result2._limbs, num_bytes); 
     } else {
        printf("\n MATCH: %d\n", i);
     }
  }

  free(instance_array);
  cudaFree(gpuInstances);
  return res_vector;
}


