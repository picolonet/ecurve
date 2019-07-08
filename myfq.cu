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

typedef struct {
    mfq_t x[MyGpuParams::BI_LIMBS];  
} single_mfq_ti;  // ti for instance, that is full array

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

__device__ __forceinline__ uint64_t addc_u64(uint64_t a, uint64_t b) {
  uint64_t r;

  asm volatile ("addc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
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

__device__ __forceinline__ uint64_t madlo_cc_u64(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;

  asm volatile ("mad.lo.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
}

__device__ __forceinline__ uint64_t madhic_u64(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;

  asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
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

__device__ __forceinline__
void dev_mul_by_2(thread_context_t& tc, mfq_t& a, mfq_t& m) {
  uint64_t temp = __shfl_up_sync(tc.sync_mask, a, 1);
  const uint64_t msb = 0x8000000000000000ull;
  a = a << 1;
  a = a | ((temp & msb) ? 0x01 : 0x0); 
  one_mod_u64(tc, a, m);
}

__device__ __forceinline__ static int32_t resolve_add_u64(thread_context_t& tc, const int64_t carry, uint64_t &x) {
  //uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
  uint32_t g, p, c;
  uint64_t sum, c;

  c=__shfl_up_sync(tc.sync_mask, carry, 1);
  c=(tc.lane ==0) ? 0 : c;
  x=add_cc_u64(x, c);
  c=addc_u64(0, 0);

  // TODO TODO TODO
  g=__ballot_sync(sync, c==1);
  p=__ballot_sync(sync, x==0xFFFFFFFF);

  sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
  c=lane&(p^sum);
  x=x+(c!=0);

  c=carry+(sum>>32);
  return __shfl_sync(sync, c, 31);
}

__device__
void fq_mul_const_mod_slow(thread_context_t& tc, mfq_t& a, mfq_t& m, uint32_t mul_const) {
  uint64_t temp_a = 0;
  for (int i = 0; i< mul_const; i++) {
    fq_add_mod(tc, temp_a, a, m); 
  }
  a = temp_a;
}

__device__
void fq_mul_const_mod(thread_context_t& tc, mfq_t& a, mfq_t& m, uint32_t mul_const) {
  uint32_t bit = 0;
  uint32_t mulc = mul_const;
  uint64_t temp_a, temp_exp_a = a;
  temp_a = (mulc & 0x01) ? a : 0;
  mulc = mulc >> 1;
  while (mulc > 0) {
    fq_add_mod(tc, temp_exp_a, temp_exp_a, m); 
    bit = mulc & 0x01;
    if (bit) fq_add_mod(tc, temp_a, temp_exp_a, m); 
    mulc= mulc >> 1;
  }
  a = temp_a;
}

__device__
void fq_mul_const_mod_fast(thread_context_t& tc, mfq_t& a, mfq_t& m, uint32_t mul_const) {
  uint32_t bit = 0;
  uint32_t mulc = mul_const;
  uint64_t temp_a, temp_exp_a = a;
  temp_a = (mulc & 0x01) ? a : 0;
  mulc = mulc >> 1;
  while (mulc > 0) {
    dev_mul_by_2(tc, temp_exp_a, m); 
    bit = mulc & 0x01;
    if (bit) fq_add_mod(tc, temp_a, temp_exp_a, m); 
    mulc= mulc >> 1;
  }
  a = temp_a;
}

__device__
void fq_sub_mod(thread_context_t& tc, mfq_t& a, mfq_t& b, mfq_t& m) {
  int which = dev_sub_u64(tc, a, b);
  if (which == -1) {
     fq_add_nomod(tc, a, m);
  }
}

__device__
uint32_t dev_mul_by_const(thread_context_t& tc, uint64_t& r, uint64_t& a, uint64_t f) {
  uint64_t carry = 0;
  uint64_t prd;
  uint32_t lane = tc.lane;
  prd = madlo_cc_u64(a, f, 0);
  carry = madhic_u64(a, f, 0);
  carry = resolve_add_u64(carry, prd);
  r = prd;
  return carry;
}

__device__ __forceinline__
void mont_mul_64(uint64_t a[], uint64_t x[], uint64_t y[], uint64_t m[], uint64_t inv, int n) {
  const uint32_t sync=0xFFFFFFFF;
  uint32_t lane = threadIdx.x % TPI;
  uint32_t ui, carry;
  uint32_t temp = 0, temp2 = 0;
  uint32_t temp_carry = 0, temp_carry2 = 0;
  uint32_t my_lane_a;

  for (int i = 0; i < n; i ++) {
     ui = madlo_cc(x[i], y[0], a[0]);
     ui = madlo_cc(ui, inv, 0);
     temp_carry = dev_mul_by_const(temp, y, x[i]);
     temp_carry2 = dev_mul_by_const(temp2, m, ui);

     temp_carry = temp_carry + dev_add_ab2(temp2, temp);
     temp_carry = temp_carry + dev_add_ab2(a[lane], temp2);

     // missing one BIG add.
     add_extra_ui32(a[lane], temp_carry, temp_carry2);
 
     // right shift one limb
     my_lane_a = a[lane];
     a[lane] =__shfl_down_sync(sync, my_lane_a, 1, TPI);
     a[lane] = (lane == (TPI -1)) ? 0 : a[lane];
  }

  // compare and subtract.
  uint32_t dummy_a = a[lane];
  int which = dev_sub(dummy_a, m[lane]);
  a[lane] = (which == -1) ? a[lane] : dummy_a; 
}

__device__
void fq_mont_mul_mod(thread_context_t& tc, mfq_t& a, mfq_t& b, mfq_t& m) {
  uint64_t sum, carry;
  sum = add_cc_u64(a, b);
  carry = addc_cc(0, 0);
  fast_propagate_add_u64(tc, carry, sum);
  a = sum;

  // DO THE MODULUS.
  one_mod_u64(tc, a, m);
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

// X * Y in mont.
__global__
void fq_mont_mul_kernel(tuple_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[]) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  fq_mont_mul_mod(tc, instances[tc.instance_number].x[tc.lane],
       instances[tc.instance_number].y[tc.lane], mnt4_modulus_device[tc.lane]);
}

__global__
void fq_mul_const_kernel(single_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[], uint32_t mul_const) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  fq_mul_const_mod_fast(tc, instances[tc.instance_number].x[tc.lane], mnt4_modulus_device[tc.lane], mul_const);
}

void load_mnt4_modulus() {
  cudaMemcpyToSymbol(mnt4_modulus_device, mnt4_modulus, bytes_per_elem, 0, cudaMemcpyHostToDevice);
}
