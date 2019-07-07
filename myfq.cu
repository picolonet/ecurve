#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <cassert>
#include "cgbn/cgbn.h"
#include "utility/support.h"

#include "myfq.h"

#define TPI 16
#define BITS 1024 
#define TPB 128    // threads per block (divible by 32)

static const uint32_t TPI_ONES=(1ull<<TPI)-1;


__device__ void compute_context(thread_context_t& t) {
   
  t.instance_number =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  t.lane = threadIdx.x % TPI;
  t.warp_number = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

  // two sub warps per warp.
  t.sync_mask = ((instance_number % 2) == 0) ? 0x0000FFFF: 0xFFFF0000;
}

__device__ __forceinline__ uint64_t add_cc_u64(uint64_t a, uint64_t b) {
  uint64_t r;

  asm volatile ("add.cc.u64 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ static int32_t fast_propagate_add_u64(thread_context& tc,
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

