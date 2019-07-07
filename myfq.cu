#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <cassert>
#include "cgbn/cgbn.h"
#include "utility/support.h"

#define TPI 16
#define BITS 1024 
#define TPB 128    // threads per block (divible by 32)

static const uint32_t TPI_ONES=(1ull<<TPI)-1;

// List of Gpu params. BI generally stands for big integer.
struct MyGpuParams {

  static const int BI_BITS = 1024;

  static const int BI_BYTES = 128; 

  static const int BI_BITS_PER_LIMB = 64; 
   
  static const int BI_LIMBS = 16;

  static const int BI_TPI = 16;  // Threads per instance, this has to match LIMBS per BigInt
};

// Fq really represents a biginteger of BI_LIMBS of type uint64_t. But since this is in
// CUDA, and gets parallely executed the class represents a single limb.
typedef struct MyFq {
    uint64_t &val;
} mfq_t;

// Class represents a big integer vector. But since it uses a GPU, all operations are
// defined on a single big integer which is of a fixed size.
// The basic data type is kept fixed at uint64_t.
typedef struct {
    mfq_t a0[MyGpuParams::BI_LIMBS];  
    mfq_t a1[MyGpuParams::BI_LIMBS];  
} mfq2_t;

typedef struct {
  mfq2_t A;
  mfq2_t B;
} mquad_t;


typedef struct {
  uint32_t lane;
  uint32_t sync_mask;
  uint32_t instance_number;
  uint32_t warp_number;
} thread_context_t;

__device__ void fq2_add(thread_context_t& tc, mfq2_t& a, mfq2_t& b);
__device__ __forceinline__ static int32_t fast_propagate_add_u64(thread_context_t& tc,
      const uint32_t carry, uint64_t &x);

__device__ void compute_context(thread_context_t& t) {
   
  t.instance_number =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  t.lane = threadIdx.x % TPI;
  t.warp_number = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

  // two sub warps per warp.
  t.sync_mask = ((t.instance_number % 2) == 0) ? 0x0000FFFF: 0xFFFF0000;
}

__device__ __forceinline__ uint64_t add_cc_u64(uint64_t a, uint64_t b) {
  uint64_t r;

  asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}

__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ static int32_t fast_propagate_add_u64(thread_context_t& tc,
      const uint32_t carry, uint64_t &x) {
    //uint32_t warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t lane_mask = 1 << tc.lane;
    uint32_t g, p, c;
    uint64_t sum;
  
    g=__ballot_sync(tc.sync_mask, carry==1);
    p=__ballot_sync(tc.sync_mask, x==0xFFFFFFFFFFFFFFFFULL);
 
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane_mask&(p^sum);
    
    x=x+(c!=0);
     
    return sum>>32;   // -(p==0xFFFFFFFF);
}

__device__
void fq2_add(mfq2_t& a, mfq2_t& b) {
  uint64_t sum, carry;
  sum = add_cc_u64(a.val, b.val);
  carry = addc_cc(0, 0);
  fast_propagate_add_u64(carry, sum);
  a.val = sum;
}

__global__
void fq2_add_kernel(quad_t* instances, uint32_t instance_count) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc);

  fq2_add(tc, instances[tc.instance_number].A[tc.lane],
              instances[tc.instance_number].B[tc.lane]);
}

