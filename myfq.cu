#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <cassert>
#include "cgbn/cgbn.h"
#include "utility/support.h"

#include "constants.h"

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

const uint64_t MNT4_INV = 0xf2044cfbe45e7fff;
const uint64_t MNT6_INV = 0xc90776e23fffffff;

const uint32_t MNT4_INV32 = 0xe45e7fff;
const uint64_t MSB_MASK = 0x8000000000000000ULL;

typedef struct {
    mfq_t x[MyGpuParams::BI_LIMBS];  
    int carry;
} single_mfq_ti;  // ti for instance, that is full array

// Class represents a big integer vector. But since it uses a GPU, all operations are
// defined on a single big integer which is of a fixed size.
// The basic data type is kept fixed at uint64_t.
typedef struct {
    mfq_t x[MyGpuParams::BI_LIMBS];  
    mfq_t y[MyGpuParams::BI_LIMBS];  
    int carry;
} tuple_mfq_ti;  // ti for instance, that is full array

typedef struct {
    mfq_t x[MyGpuParams::BI_LIMBS];  
    mfq_t y[MyGpuParams::BI_LIMBS];  
    mfq_t r[MyGpuParams::BI_LIMBS];  
} triple_mfq_ti;  // ti for instance, that is full array

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

__device__ mfq_t mfq_zero() { return 0;}

__device__ mfq_t mfq_one(thread_context_t& tc) { 
   return ((uint64_t*)mnt4_const_one)[tc.lane];
}

__device__ void fq2_add(thread_context_t& tc, mfq_t& a, mfq_t& b);

__device__ __forceinline__ static uint64_t fast_propagate_add_u64(thread_context_t& tc,
      const uint64_t carry, uint64_t &x);

__device__ void compute_context(thread_context_t& t, uint32_t instance_count) {
   
  t.instance_number =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  t.lane = threadIdx.x & TPI-1;
  t.warp_number = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  t.instance_count = instance_count;

  // two sub warps per warp.
  t.subwarp_number = t.instance_number % 2;
  t.sync_mask = (t.subwarp_number == 0) ? 0x0000FFFF: 0xFFFF0000;
}

__device__
mfq_t get_const_limb(thread_context_t& tc, uint64_t limb) {
  return (tc.lane == 0) ? limb : 0;
}

// Returns true if the multi-precision x is equal to 00..00limb
__device__
bool is_equal_limb(thread_context_t& tc, mfq_t x, uint64_t limb) {
  uint64_t val = (tc.lane == 0) ? limb : 0;
  uint32_t g = __ballot_sync(tc.sync_mask, x==val);
  return g == tc.sync_mask;
}

__device__
void right_shift_bits(thread_context_t& tc, mfq_t& x, int num_bits) {
  uint32_t lsb_thread = (tc.subwarp_number == 0) ? 0 : 16;
  uint32_t limb = __shfl_down_sync(tc.sync_mask, x, 1);
  x = x >> 1;
  uint64_t bit = limb & 0x01;
  if (bit && (tc.lane != 15)) {
    x = x | MSB_MASK;
  }
}

// Returns true if the multi-precision x is equal to 00..00limb
__device__
bool is_even(thread_context_t& tc, mfq_t x) {
  uint32_t lsb_thread = (tc.subwarp_number == 0) ? 0 : 16;
  uint64_t limb = __shfl_sync(tc.sync_mask, x, lsb_thread) ;
  return (limb & 0x01) == 0;
}

__device__ __forceinline__ 
bool is_zero(thread_context_t& tc, mfq_t x) {
  uint32_t g = __ballot_sync(tc.sync_mask, x==0);
  return g == tc.sync_mask;
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

__device__ __forceinline__ uint64_t addc_cc_u64(uint64_t a, uint64_t b) {
  uint64_t r;

  asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}

__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ static uint64_t fast_propagate_add_u64(thread_context_t& tc,
      const uint64_t carry, uint64_t &x) {
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

__device__ __forceinline__ static int32_t fast_propagate_sub_u64(thread_context_t& tc, const uint64_t carry, uint64_t &x) {
    // uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t lane_mask = 1 << tc.lane;
    uint32_t g, p, c;
    uint32_t sum = 0;

    g=__ballot_sync(tc.sync_mask, carry==0xFFFFFFFFFFFFFFFFull);
    p=__ballot_sync(tc.sync_mask, x==0);

    g = (tc.subwarp_number == 0) ? g : (g >> 16);
    p = (tc.subwarp_number == 0) ? p : (p >> 16);
    //sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    sum=g+g+p;
    c=lane_mask&(p^sum);

    x=x-(c!=0);
    return (sum>>16) & 0x01;     // -(p==0xFFFFFFFF);
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
   uint64_t carry;
   a = sub_cc_u64(a, b);
   carry=subc_u64(0, 0);
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
int linear_mod_u64_helper(thread_context_t& tc, uint64_t& a, uint64_t& b) {
  uint64_t dummy_a = a;
  int which = dev_sub_u64(tc, dummy_a, b);
  a = (which == -1) ? a : dummy_a; 
  return which;
}

__device__ 
void linear_mod_u64(thread_context_t& tc, uint64_t& a, uint64_t& b) {
   int which = 0;
   uint64_t dummy_a = a;
   while (which >=0) {
      which = linear_mod_u64_helper(tc, dummy_a, b);
      __syncthreads();
   }
   a = dummy_a;
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

__device__ __forceinline__ static uint64_t resolve_add_u64(thread_context_t& tc, const int64_t carry, uint64_t &x) {
  //uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
  uint32_t g, p;
  uint32_t lane_mask = 1 << tc.lane;
  uint64_t sum, c;

  c=__shfl_up_sync(tc.sync_mask, carry, 1);
  c=(tc.lane ==0) ? 0 : c;
  x=add_cc_u64(x, c);
  c=addc_u64(0, 0);

  g=__ballot_sync(tc.sync_mask, c==1);
  p=__ballot_sync(tc.sync_mask, x==0xFFFFFFFFFFFFFFFFull);

  g = (tc.subwarp_number == 0) ? g : g >> 16;
  p = (tc.subwarp_number == 0) ? p : p >> 16;
  sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
  uint32_t c1=lane_mask&(p^sum);
  x=x+(c1!=0);

  c=carry+(sum>>16);
  uint32_t top_thread = (tc.subwarp_number == 0) ? 15 : 31;
  return __shfl_sync(tc.sync_mask, c, top_thread);
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

// TODO: Get rid of a[] and pass in the lane variable.
__device__ __forceinline__
uint64_t dev_mul_by_const(thread_context_t& tc, uint64_t& r, uint64_t a[], uint64_t f) {
  uint64_t carry = 0;
  uint64_t prd;
  uint32_t lane = tc.lane;
  prd = madlo_cc_u64(a[lane], f, 0);
  carry = madhic_u64(a[lane], f, 0);
  carry = resolve_add_u64(tc, carry, prd);
  r = prd;
  return carry;
}

__device__ __forceinline__
uint64_t dev_mul_by_const_lane(thread_context_t& tc, uint64_t& r, uint64_t& a, uint64_t f) {
  uint64_t carry = 0;
  uint64_t prd;
  uint32_t lane = tc.lane;
  prd = madlo_cc_u64(a, f, 0);
  carry = madhic_u64(a, f, 0);
  carry = resolve_add_u64(tc, carry, prd);
  r = prd;
  return carry;
}

__device__ __forceinline__
uint64_t dev_add_ab2(thread_context_t& tc, uint64_t& a, uint64_t b) {
  uint64_t sum, carry;
  sum = add_cc_u64(a, b);
  carry = addc_cc_u64(0, 0);
  carry = fast_propagate_add_u64(tc, carry, sum);

  a = sum;
  
  // a[TPI] = a[TPI] + carry + b_msb_carry;
  return carry;
}

__device__ __forceinline__
uint64_t add_extra_u64(thread_context_t& tc, uint64_t& a,
         const uint64_t extra, const uint64_t extra_carry) {
  uint64_t sum, carry, result;
  //uint32_t group_thread=threadIdx.x & TPI-1;

  sum = add_cc(a, (tc.lane == 0) ? extra : 0);
  carry = addc_cc(0, (tc.lane == 0) ? extra_carry : 0);

  // Each time we call fast_propagate_add, we might have to "clear_carry()"
  // to clear extra data when Padding threads are used.
  result=fast_propagate_add_u64(tc, carry, sum);
  a = sum;
  return result;
}

__device__
uint64_t fq_mul_const_mod_test(thread_context_t& tc, mfq_t& a, mfq_t& m, uint64_t f) {
  uint64_t carry = 0;
  uint64_t prd;
  uint32_t lane = tc.lane;
  prd = madlo_cc_u64(a, f, 0);
  carry = madhic_u64(a, f, 0);
  carry = resolve_add_u64(tc, carry, prd);
  a = prd;
  //linear_mod_u64(tc, a, m);
  //one_mod_u64(tc, a, m);
  return carry;
}

// THIS will ONLY work when n is 12 as we use shared memory. 
__device__ 
void mont_mul_64_lane(thread_context_t& tc, uint64_t& a_limb, uint64_t& x_limb,
      uint64_t& y_limb, uint64_t m[], uint64_t inv, int n) {
  //const uint32_t sync=0xFFFFFFFF;
  //uint32_t lane = threadIdx.x % TPI;
  uint64_t ui, carry;
  uint64_t temp = 0, temp2 = 0;
  uint64_t temp_carry = 0, temp_carry2 = 0;
  uint64_t my_lane_a;
  uint64_t temp_carry3 = 0, temp_carry4 = 0;
  uint64_t y_0 = __shfl_sync(tc.sync_mask, y_limb, tc.subwarp_number * 16);

  a_limb = 0;

  for (int i = 0; i < n; i ++) {
     uint64_t x_i = __shfl_sync(tc.sync_mask, x_limb, i + tc.subwarp_number * 16);
     uint64_t a_0 = __shfl_sync(tc.sync_mask, a_limb, tc.subwarp_number * 16);
     // ui = madlo_cc_u64(x[i], y[0], a[0]);
     ui = madlo_cc_u64(x_i, y_0, a_0);
     ui = madlo_cc_u64(ui, inv, 0);
     temp_carry = dev_mul_by_const_lane(tc, temp, y_limb, x_i);
     temp_carry2 = dev_mul_by_const_lane(tc, temp2, m[tc.lane], ui); 

     temp_carry3 =  dev_add_ab2(tc, temp2, temp);
     temp_carry4 = dev_add_ab2(tc, a_limb, temp2);

     //temp_carry3 =  fq_add_nomod(tc, temp2, temp);
     //temp_carry4 = fq_add_nomod(tc, a[tc.lane], temp2);

     //temp_carry = add_cc(temp_carry,  temp_carry3);
     //temp_carry = addc_cc(temp_carry, temp_carry4);

     //temp_carry = add_cc_u64(temp_carry, temp_carry2);

     // missing one BIG add.
     //add_extra_u64(tc, a[tc.lane], temp_carry, 0);
 
     // right shift one limb
     my_lane_a = a_limb;
     a_limb =__shfl_down_sync(tc.sync_mask, my_lane_a, 1);
     a_limb = (tc.lane == (TPI - 1)) ? 0 : a_limb;
  }

  // compare and subtract. Essentially the logic of:
  // one_mod_u64(tc, a[tc.lane], m[tc.lane]);
  uint64_t dummy_a = a_limb;
  int which = dev_sub_u64(tc, dummy_a, m[tc.lane]);
  a_limb = (which == -1) ?  a_limb : dummy_a; 
}

__device__ 
void mont_mul_64(thread_context_t& tc, uint64_t a[], uint64_t x[],
      uint64_t y[], uint64_t m[], uint64_t inv, int n) {
  //const uint32_t sync=0xFFFFFFFF;
  //uint32_t lane = threadIdx.x % TPI;
  uint64_t ui, carry;
  uint64_t temp = 0, temp2 = 0;
  uint64_t temp_carry = 0, temp_carry2 = 0;
  uint64_t my_lane_a;
  uint64_t temp_carry3 = 0, temp_carry4 = 0;

  a[tc.lane] = 0;

  for (int i = 0; i < n; i ++) {
     ui = madlo_cc_u64(x[i], y[0], a[0]);
     ui = madlo_cc_u64(ui, inv, 0);
     temp_carry = dev_mul_by_const(tc, temp, y, x[i]);
     temp_carry2 = dev_mul_by_const(tc, temp2, m, ui); 

     temp_carry3 =  dev_add_ab2(tc, temp2, temp);
     temp_carry4 = dev_add_ab2(tc, a[tc.lane], temp2);

     //temp_carry3 =  fq_add_nomod(tc, temp2, temp);
     //temp_carry4 = fq_add_nomod(tc, a[tc.lane], temp2);

     //temp_carry = add_cc(temp_carry,  temp_carry3);
     //temp_carry = addc_cc(temp_carry, temp_carry4);

     //temp_carry = add_cc_u64(temp_carry, temp_carry2);

     // missing one BIG add.
     //add_extra_u64(tc, a[tc.lane], temp_carry, 0);
 
     // right shift one limb
     my_lane_a = a[tc.lane];
     a[tc.lane] =__shfl_down_sync(tc.sync_mask, my_lane_a, 1);
     a[tc.lane] = (tc.lane == (TPI - 1)) ? 0 : a[tc.lane];
  }

  // compare and subtract.
  one_mod_u64(tc, a[tc.lane], m[tc.lane]);
}

// Computes the inverse of a mod m
__device__
mfq_t fq_inverse(thread_context_t& tc, mfq_t&a, mfq_t& m) {
  mfq_t u = a;
  mfq_t v = m;

  mfq_t x1 = get_const_limb(tc, 1);  // x1 = 1;
  mfq_t x2 = 0;

  while (!is_equal_limb(tc, u, 1) && !is_equal_limb(tc, v, 1)) {

    while (is_even(tc, u)) {
      right_shift_bits(tc, u, 1);
      if (is_even(tc, x1)) {
        right_shift_bits(tc, x1, 1);
      } else {
        fq_add_nomod(tc, x1, m);
        right_shift_bits(tc, x1, 1);
      }
    }
    while (is_even(tc, v)) {
      right_shift_bits(tc, v, 1);
      if (is_even(tc, x2)) {
        right_shift_bits(tc, x2, 1);
      } else {
        fq_add_nomod(tc, x2, m);
        right_shift_bits(tc, x2, 1);
      }
    }

    mfq_t tmp_u = u;
    int which = dev_sub_u64(tc, tmp_u, v);
    if (which == -1) {
       dev_sub_u64(tc, v, u);
       dev_sub_u64(tc, x2, x1); 
    } else {
      u = tmp_u;
      dev_sub_u64(tc, x1, x2); 
    }
  }
  if (is_equal_limb(tc, u, 1)) {
    //linear_mod_u64(tc, x1, m);
    return x1;
  } else {
    //linear_mod_u64(tc, x2, m);
    return x2;
  } 
}

// THIS IS WRONG
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

__device__
mfq_t myfq_square(thread_context_t& tc, mfq_t& limb) {
  uint64_t result_limb;
  mont_mul_64_lane(tc, result_limb, limb,
       limb, mnt4_modulus_device, MNT4_INV, 12);
  return result_limb;
}

////////////////////////////////////////////////////
// GLOBAL ROUTINES.
////////////////////////////////////////////////////
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
// X - Y
__global__
void fq_sub_nomod_kernel(tuple_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[]) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  instances[tc.instance_number].carry =
      dev_sub_u64(tc, instances[tc.instance_number].x[tc.lane],
      instances[tc.instance_number].y[tc.lane]);
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
void fq_mont_mul_kernel(triple_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[]) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  // The 12 number is key, this is because we have 12 limbs and the Rinv is calculated for that
  // in order to match the data generated by the reference libff implementation.
  mont_mul_64(tc, instances[tc.instance_number].r, instances[tc.instance_number].x,
       instances[tc.instance_number].y, mnt4_modulus_device, MNT4_INV, 12);
}

__global__
void fq_mont_mul_lane_kernel(triple_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[]) {
  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  // The 12 number is key, this is because we have 12 limbs and the Rinv is calculated for that
  // in order to match the data generated by the reference libff implementation.
  mont_mul_64_lane(tc, instances[tc.instance_number].r[tc.lane], instances[tc.instance_number].x[tc.lane],
       instances[tc.instance_number].y[tc.lane], mnt4_modulus_device, MNT4_INV, 12);
}

__global__
void fq_inverse_kernel(single_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[]) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  //fq_mul_const_mod_fast(tc, instances[tc.instance_number].x[tc.lane], mnt4_modulus_device[tc.lane], mul_const);
  // instances[tc.instance_number].carry = 
  // fq_mul_const_mod_test(tc, instances[tc.instance_number].x[tc.lane], mnt4_modulus_device[tc.lane], mul_const);
  instances[tc.instance_number].x[tc.lane] =
     fq_inverse(tc, instances[tc.instance_number].x[tc.lane], mnt4_modulus_device[tc.lane]);
 // mfq_t tmp_m = mnt4_modulus_device[tc.lane];
 // linear_mod_u64(tc, tmp_m, instances[tc.instance_number].x[tc.lane]);
 // instances[tc.instance_number].x[tc.lane] = tmp_m;
}

__global__
void fq_mul_const_kernel(single_mfq_ti* instances, uint32_t instance_count, mfq_t modulus[], uint64_t mul_const) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  if(my_instance>=instance_count) return;    // return if my_instance is not valid

  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;

  //fq_mul_const_mod_fast(tc, instances[tc.instance_number].x[tc.lane], mnt4_modulus_device[tc.lane], mul_const);
  // instances[tc.instance_number].carry = 
  // fq_mul_const_mod_test(tc, instances[tc.instance_number].x[tc.lane], mnt4_modulus_device[tc.lane], mul_const);
  fq_mul_const_mod(tc, instances[tc.instance_number].x[tc.lane], mnt4_modulus_device[tc.lane], mul_const);
}

void load_mnt4_modulus() {
  cudaMemcpyToSymbol(mnt4_modulus_device, mnt4_modulus, bytes_per_elem, 0, cudaMemcpyHostToDevice);
}
