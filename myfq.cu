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
typedef uint64_t mfq_t;

__constant__ mfq_t mnt4_modulus_device[16];

// Class represents a big integer vector. But since it uses a GPU, all operations are
// defined on a single big integer which is of a fixed size.
// The basic data type is kept fixed at uint64_t.
typedef struct {
    mfq_t x[MyGpuParams::BI_LIMBS];  
    mfq_t y[MyGpuParams::BI_LIMBS];  
} tuple_mfq_ti;  // ti for instance, that is full array

typedef struct {
    mfq_t a0[MyGpuParams::BI_LIMBS];  
    mfq_t a1[MyGpuParams::BI_LIMBS];  
} mfq2_ti;  // ti for instance, that is full array

typedef struct {
  mfq2_ti A;
  mfq2_ti B;
} mquad_ti;

typedef struct {
  uint32_t lane;
  uint32_t sync_mask;
  uint32_t instance_number;
  uint32_t instance_count;
  uint32_t warp_number;
  uint32_t subwarp_number; // 0 or 1
} thread_context_t;

__device__ void fq2_add(thread_context_t& tc, mfq_t& a, mfq_t& b);

__device__ __forceinline__ static int32_t fast_propagate_add_u64(thread_context_t& tc,
      const uint32_t carry, uint64_t &x);

__device__ void compute_context(thread_context_t& t, uint32_t instance_count) {
   
  t.instance_number =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  t.lane = threadIdx.x & TPI-1;
  t.warp_number = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  t.instance_count = instance_count;

  // two sub warps per warp.
  t.subwarp_number = t.instance_number % 2;
  t.sync_mask = (t.subwarp_number == 0) ? 0x0000FFFF: 0xFFFF0000;
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

__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ static int32_t fast_propagate_add_u64(thread_context_t& tc,
      const uint32_t carry, uint64_t &x) {
    //uint32_t warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t lane_mask = 1 << tc.lane;
    uint32_t g, p, c;
    uint64_t sum;
  
    g=__ballot_sync(tc.sync_mask, carry==1);
    p=__ballot_sync(tc.sync_mask, x==0xFFFFFFFFFFFFFFFFull);

    g = (tc.subwarp_number == 0) ? g : g >> 16;
    p = (tc.subwarp_number == 0) ? p : p >> 16;
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane_mask&(p^sum);
    
    x=x+(c!=0);
     
    return sum>>16;   // -(p==0xFFFFFFFF);
}

__device__
void fq2_add_nomod(thread_context_t& tc, mfq_t& a, mfq_t& b) {
  uint64_t sum, carry;
  // THIS IS WRONG. FIX ME.
  sum = add_cc_u64(a, b);
  carry = addc_cc(0, 0);
  fast_propagate_add_u64(tc, carry, sum);
  a = sum;
}

__device__
void fq2_add(thread_context_t& tc, mfq_t& a, mfq_t& b) {
   // HUGELY WRONG. FIX ME.
   fq2_add_nomod(tc, a, b);
}

__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ static int32_t fast_propagate_sub_u64(thread_context_t& tc, const uint32_t carry, uint64_t &x) {
    // uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t lane_mask = 1 << tc.lane;
    uint32_t g, p, c;
    uint64_t sum = 0;

    g=__ballot_sync(tc.sync_mask, carry==0xFFFFFFFF);
    p=__ballot_sync(tc.sync_mask, x==0);

    g = (tc.subwarp_number == 0) ? g : (g >> 16);
    p = (tc.subwarp_number == 0) ? p : (p >> 16);
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane_mask&(p^sum);

    x=x-(c!=0);
    return (sum>>16);     // -(p==0xFFFFFFFF);
}

__device__ __forceinline__ static int32_t fast_propagate_sub(thread_context_t& tc, const uint32_t carry, uint32_t &x) {
    // uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
    uint32_t lane_mask = 1 << tc.lane;
  
    g=__ballot_sync(tc.sync_mask, carry==0xFFFFFFFF);
    p=__ballot_sync(tc.sync_mask, x==0);

    g = (tc.subwarp_number == 0) ? g : g >> 16;
    p = (tc.subwarp_number == 0) ? p : p >> 16;
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane_mask&(p^sum);

    x=x-(c!=0);
    return (sum>>32);     // -(p==0xFFFFFFFF);
}

__device__ __forceinline__ uint64_t subc_u64(uint64_t a, uint64_t b) {
  uint64_t r;

  asm volatile ("subc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}

__device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint64_t sub_cc_u64(uint64_t a, uint64_t b) {
  uint64_t r;

  asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}

__device__
int dev_sub(thread_context_t& tc, uint32_t& a, uint32_t& b) {
   uint32_t carry;
   a = sub_cc(a, b);
   carry=subc(0, 0);
   return -fast_propagate_sub(tc, carry, a); 
}

__device__ __forceinline__
int dev_sub_u64(thread_context_t& tc, uint64_t& a, uint64_t& b) {
   uint32_t carry;
   a = sub_cc_u64(a, b);
   carry=subc(0, 0);
   return -fast_propagate_sub_u64(tc, carry, a); 
}

// Assuming either a < b or a > b and a < 2b. we subtract b
// from a and test.
__device__ __forceinline__
void one_mod_u64(thread_context_t& tc, uint64_t& a, uint64_t& b) {
  uint64_t dummy_a = a;
  int which = dev_sub_u64(tc, dummy_a, b);
  a = (which == -1) ? a : dummy_a; 
}

__device__
int32_t fq_add_nomod(thread_context_t& tc, mfq_t& a, mfq_t& b) {
  uint64_t sum, carry;
  sum = add_cc_u64(a, b);
  carry = addc_cc(0, 0);
  carry = fast_propagate_add_u64(tc, carry, sum);
  a = sum;
  return carry;
}

__device__
void fq_add_mod(thread_context_t& tc, mfq_t& a, mfq_t& b, mfq_t& m) {
  uint64_t sum, carry;
  sum = add_cc_u64(a, b);
  carry = addc_cc(0, 0);
  fast_propagate_add_u64(tc, carry, sum);
  a = sum;

  // DO THE MODULUS.
  one_mod_u64(tc, a, m);
}

__device__
void fq_sub_mod(thread_context_t& tc, mfq_t& a, mfq_t& b, mfq_t& m) {
  int which = dev_sub_u64(tc, a, b);
  if (which == -1) {
     fq_add_nomod(tc, a, m);
  }
}

__global__
void fq2_add_kernel(mquad_ti* instances, uint32_t instance_count) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  // THIS IS WRONG.
  fq2_add(tc, instances[tc.instance_number].A.a0[tc.lane],
             instances[tc.instance_number].B.a0[tc.lane]);
}

// X - Y
__global__
void fq_sub_kernel(tuple_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[]) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  fq_sub_mod(tc, instances[tc.instance_number].x[tc.lane],
             instances[tc.instance_number].y[tc.lane], mnt4_modulus_device[tc.lane]);
}

__global__
void fq_add_kernel(tuple_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[]) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  fq_add_mod(tc, instances[tc.instance_number].x[tc.lane],
             instances[tc.instance_number].y[tc.lane], mnt4_modulus_device[tc.lane]);
}

void load_mnt4_modulus() {
  cudaMemcpyToSymbol(mnt4_modulus_device, mnt4_modulus, bytes_per_elem, 0, cudaMemcpyHostToDevice);
}
