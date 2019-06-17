#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "utility/support.h"


#define TPI 32
#define BITS 1024

#define TPB 32    // the number of threads per block to launch (must be divisible by 32

typedef struct {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> y;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> result;
} instance_t;


typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, 1024> env1024_t;

__global__ void add_kernel(instance_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context;                                 // create a CGBN context
  env1024_t         bn1024_env(bn_context);                     // construct a bn environment for 1024 bit math
  env1024_t::cgbn_t a, b, r, r2, m;                             // three 1024-bit values (spread across a warp)
  uint32_t np0;
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
  
  cgbn_load(bn1024_env, a, &(problem_instances[my_instance]).x);
  cgbn_load(bn1024_env, b, &(problem_instances[my_instance]).y);
  cgbn_load(bn1024_env, m, &(problem_instances[my_instance]).m);
  
  // cgbn_add(bn1024_env, r, a, b);
  np0 = -cgbn_binary_inverse_ui32(bn1024_env, cgbn_get_ui32(bn1024_env, m));
  // np0=cgbn_bn2mont(bn1024_env, a, a, m);
  // cgbn_bn2mont(bn1024_env, b, b, m);
  cgbn_mont_mul(bn1024_env, r, a, b, m, np0);
  //cgbn_mont2bn(bn1024_env, r, r, m, np0);
  // int use_r2 = cgbn_sub(bn1024_env, r2, r, m);
  int use_r2 = -1;
  
  if (use_r2 == 0) {
   cgbn_store(bn1024_env, &(problem_instances[my_instance].result), r2);
  } else {
   cgbn_store(bn1024_env, &(problem_instances[my_instance].result), r);
  }
}


uint8_t* call_mycuda(uint8_t* x, uint8_t* y, uint8_t *m, int num_bytes) {
  int count = 1;
  instance_t *gpuInstances;
  instance_t* instance_array = (instance_t*) malloc(sizeof(instance_t) * count);
  cgbn_error_report_t *report;

  // create a cgbn_error_report for CGBN to report back errors
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  uint32_t* x32 = (uint32_t*) x;
  uint32_t* y32 = (uint32_t*) y;
  uint32_t* m32 = (uint32_t*) m;

  for (int i = 0; i < num_bytes / 4; i ++ ) {
     instance_array->x._limbs[i] = x32[i];
  }

  for (int i = 0; i < num_bytes / 4; i ++ ) {
     instance_array->y._limbs[i] = y32[i];
  }

  for (int i = 0; i < num_bytes / 4; i ++ ) {
     instance_array->m._limbs[i] = m32[i];
  }

  printf("Copying instances to the GPU ...\n");
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*count));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(instance_t)*count, cudaMemcpyHostToDevice));

  add_kernel<<<1, 32>>>(gpuInstances, 1);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(instance_t)*count, cudaMemcpyDeviceToHost));

  uint8_t* result = (uint8_t*) malloc(num_bytes * sizeof(uint8_t));
  uint32_t* result32 = (uint32_t*) result;
  for (int i = 0; i < num_bytes / 4; i ++ ) {
     result32[i] = instance_array->result._limbs[i];
  }
  printf("Done. returning ...\n");
  return result;
}
