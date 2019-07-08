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

static const uint32_t TPI_ONES=(1ull<<TPI)-1;

typedef struct {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> y;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> result1;
  cgbn_mem_t<BITS> result2;
  cgbn_mem_t<BITS> mul_lo;
  cgbn_mem_t<BITS> mul_hi;
  cgbn_mem_t<2*BITS> winp;
  cgbn_mem_t<BITS> wout;
} my_instance_t;

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env1024_t;

const uint64_t MNT4_INV = 0xf2044cfbe45e7fff;
const uint64_t MNT6_INV = 0xc90776e23fffffff;

const uint32_t MNT4_INV32 = 0xe45e7fff;

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

__device__
uint32_t dev_mul_by_const(uint32_t& r, uint32_t a[], uint32_t f) {
  uint32_t carry = 0;
  uint32_t prd, lane = threadIdx.x % TPI;
  prd = madlo_cc(a[lane], f, 0);
  carry=madhic(a[lane], f, 0);
  carry=resolve_add(carry, prd);
  r = prd;
  return carry;
}

// Result is stored in r = a + b. a is of size 2n, b is of size n.
__device__
uint32_t dev_add_ab(uint32_t r[], uint32_t a[], uint32_t b[]) {
  uint32_t lane = threadIdx.x % TPI;
  uint32_t sum, carry;
  sum = add_cc(a[lane], b[lane]);
  carry = addc_cc(0, 0);
  carry = fast_propagate_add(carry, sum);

  r[lane] = sum;
  
  // a[TPI] = a[TPI] + carry + b_msb_carry;
  return carry;
}

__device__
uint32_t dev_add_ab2(uint32_t& a, uint32_t b) {
  uint32_t sum, carry;
  sum = add_cc(a, b);
  carry = addc_cc(0, 0);
  carry = fast_propagate_add(carry, sum);

  a = sum;
  
  // a[TPI] = a[TPI] + carry + b_msb_carry;
  return carry;
}

__device__
uint32_t add_extra_ui32(uint32_t& a, const uint32_t extra, const uint32_t extra_carry) {
  uint32_t sum, carry, result;
  uint32_t group_thread=threadIdx.x & TPI-1;
  sum = add_cc(a, (group_thread==0) ? extra : 0);
  carry = addc_cc(0, (group_thread==0) ? extra_carry : 0);

  // Each time we call fast_propagate_add, we might have to "clear_carry()"
  // to clear extra data when Padding threads are used.
  result=fast_propagate_add(carry, sum);
  a = sum;
}

__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

 __device__ __forceinline__ static int32_t fast_propagate_sub(const uint32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    g=__ballot_sync(sync, carry==0xFFFFFFFF);
    p=__ballot_sync(sync, x==0);

    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);

    x=x-(c!=0);
    return (sum>>32);     // -(p==0xFFFFFFFF);
  }

__device__
int dev_sub(uint32_t& a, uint32_t& b) {
   uint32_t carry = sub_cc(a, b);
   return -fast_propagate_sub(carry, a); 
}

// n has to be equal to TPI, otherwise this will blow up.
__device__
void mont_redc_wide(uint32_t a[], uint32_t m[], uint32_t inv, int n) {
  const uint32_t sync=0xFFFFFFFF;
  uint32_t lane = threadIdx.x % TPI;
  uint32_t ui, carry;
  uint32_t temp = 0, temp_carry = 0, temp_carry_2 = 0;
  uint32_t my_lane_a;

  for (int i = 0; i < n; i ++) {
     ui = madlo_cc(inv, a[0], 0);

     // temp = modulus * ui
     temp_carry = dev_mul_by_const(temp, m, ui);

     temp_carry_2 = dev_add_ab2(a[lane], temp);

     // add to A, handle carry.
     add_extra_ui32(a[lane + n], temp_carry, temp_carry_2);

     // right shift an array of size 2n.
     my_lane_a = a[lane];
     a[lane] =__shfl_down_sync(sync, my_lane_a, 1, TPI);
     a[lane] = (lane == (TPI -1)) ? a[lane+1] : a[lane];

     my_lane_a = a[lane + n];
     a[lane + n] =__shfl_down_sync(sync, my_lane_a, 1, TPI);
  }
  
  // compare and subtract.
  uint32_t dummy_a = a[lane];
  int which = dev_sub(dummy_a, m[lane]);
  a[lane] = (which == -1) ? a[lane] : dummy_a; 
}

__device__
void mont_mul(uint32_t a[], uint32_t x[], uint32_t y[], uint32_t m[], uint32_t inv, int n) {
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

// num is of size 2*n. modulus is of size n
// result is of size n.
void reduce_wide(mp_limb_t* result, mp_limb_t* num, mp_limb_t* modulus, uint64_t inv, int n) {
        mp_limb_t *res = num;
        // mp_limb_t res[2*n];
        // mpn_mul_n(res, this->mont_repr.data, other.data, n);

        /*
          The Montgomery reduction here is based on Algorithm 14.32 in
          Handbook of Applied Cryptography
          <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
         */
        for (size_t i = 0; i < n; ++i)
        {
            mp_limb_t k = inv * res[i];
            /* calculate res = res + k * mod * b^i */
            mp_limb_t carryout = mpn_addmul_1(res+i, modulus, n, k);
            carryout = mpn_add_1(res+n+i, res+n+i, n-i, carryout);
            assert(carryout == 0);
        }

        if (mpn_cmp(res+n, modulus, n) >= 0)
        {
            const mp_limb_t borrow = mpn_sub(res+n, res+n, n, modulus, n);
            assert(borrow == 0);
        }

        mpn_copyi(result, res+n, n);
}

__global__ void my_mul_const_kernel1(my_instance_t *problem_instances, uint32_t instance_count, uint32_t f) {
  int32_t my_instance =(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid

  uint32_t lane = threadIdx.x % TPI;
  uint32_t prd, carry = 0;

  cgbn_mem_t<BITS>& a = problem_instances[my_instance].x;
  cgbn_mem_t<BITS>& b = problem_instances[my_instance].y;
  cgbn_mem_t<BITS>& r = problem_instances[my_instance].result2;

  dev_mul_by_const(prd, a._limbs, f);
  // prd = madlo_cc(a._limbs[lane], f, carry);
  // carry=madhic(a._limbs[lane], f, 0);
  // carry=resolve_add(carry, prd);
  r._limbs[lane] = prd;
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

__global__ void my_wide_redc_kernel(my_instance_t *problem_instances, uint32_t instance_count) {
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
 

  uint32_t lane = threadIdx.x % TPI;
  uint32_t sum, carry;
  cgbn_mem_t<2*BITS>& a = problem_instances[my_instance].winp;
  cgbn_mem_t<BITS>& m = problem_instances[my_instance].m;
  cgbn_mem_t<BITS>& r = problem_instances[my_instance].wout;

  mont_redc_wide(a._limbs, m._limbs, MNT4_INV32, 32);
}

__global__ void my_add_kernel1(my_instance_t *problem_instances, uint32_t instance_count) {
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

__global__ 
void my_mont_mul_kernel(my_instance_t *problem_instances, uint32_t instance_count) {
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
  cgbn_mem_t<BITS>& a = problem_instances[my_instance].x;
  cgbn_mem_t<BITS>& b = problem_instances[my_instance].y;
  cgbn_mem_t<BITS>& m = problem_instances[my_instance].m;
  cgbn_mem_t<BITS>& r = problem_instances[my_instance].result2;
  mont_mul(r._limbs, a._limbs, b._limbs, m._limbs, MNT4_INV32, 32);
}

__global__ void my_mul_kernel(my_instance_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context;                                 // create a CGBN context
  env1024_t         bn1024_env(bn_context);                     // construct a bn environment for 1024 bit math
  env1024_t::cgbn_t a, b, m;                      // three 1024-bit values (spread across a warp)
  env1024_t::cgbn_wide_t mul_wide;
  uint32_t np0;
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
  
  cgbn_load(bn1024_env, a, &(problem_instances[my_instance]).x);
  cgbn_load(bn1024_env, b, &(problem_instances[my_instance]).y);
  cgbn_load(bn1024_env, m, &(problem_instances[my_instance]).m);

  np0 = -cgbn_binary_inverse_ui32(bn1024_env, cgbn_get_ui32(bn1024_env, m));
  if (threadIdx.x == 0) {
  printf("\n %08x\n", np0);
  }

  cgbn_mul_wide(bn1024_env, mul_wide, a, b);

  cgbn_store(bn1024_env, &(problem_instances[my_instance].mul_lo), mul_wide._low);
  cgbn_store(bn1024_env, &(problem_instances[my_instance].mul_hi), mul_wide._high);
}

std::vector<uint8_t*>* compute_mul_redc_cuda(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t* input_m_base, int num_bytes, uint64_t inv,
     FILE* debug_file) {
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

  printf("Copying instances to the GPU ...\n");
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(my_instance_t)*num_elements));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(my_instance_t)*num_elements, cudaMemcpyHostToDevice));
  
  int tpb = TPB;
  printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;
  int tpi = TPI;
  printf("\n Threads per instance = %d", tpi);
  printf("\n Instances per block = %d", IPB);

  uint32_t num_blocks = (num_elements+IPB-1)/IPB;
  printf("\n Number of blocks = %d", num_blocks);

  my_mul_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  //printf("Copying results back to CPU ...\n");
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(my_instance_t)*num_elements, cudaMemcpyDeviceToHost));

  int num_limbs = num_bytes / 8;
  int num_limbs_32 = num_bytes / 4;
  for (int i = 0; i < num_elements; i ++) {
    std::memcpy((void*)instance_array[i].winp._limbs, (void*)instance_array[i].mul_lo._limbs, num_bytes);
    std::memcpy((void*)(&instance_array[i].winp._limbs[num_limbs_32]), (void*)instance_array[i].mul_hi._limbs, num_bytes);
  }
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(my_instance_t)*num_elements, cudaMemcpyHostToDevice));
  my_wide_redc_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(my_instance_t)*num_elements, cudaMemcpyDeviceToHost));
  my_mont_mul_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(my_instance_t)*num_elements, cudaMemcpyDeviceToHost));

  printf("\n Setting num 64 limbs = %d", num_limbs);
  mp_limb_t* num = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs * 2);
  mp_limb_t* modulus = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);
  std::memcpy((void*) modulus, (const void*) instance_array->m._limbs, num_bytes);

  //printf("\n Dumping modulus:");
  //gmp_printf("%Nx\n", modulus, num_limbs); 

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
    // Reduce
    std::memcpy((void*)num, (const void*)instance_array[i].mul_lo._limbs, num_bytes);
    std::memcpy((void*) (num + num_limbs), (const void*)instance_array[i].mul_hi._limbs, num_bytes);
    mp_limb_t* fresult = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);
 
    // printf("\n Dumping 64 byte limb wide num [%d]:", i);
    // gmp_printf("%Nx\n", num, num_limbs * 2); 

    reduce_wide(fresult, num, modulus, inv, num_limbs);
    fprintf(debug_file, "\n MP REDC: %d\n", i);
    fprint_uint8_array(debug_file, (uint8_t*) fresult, num_bytes);

    fprintf(debug_file, "\n MYCUDA FAST REDC: %d\n", i);
    fprint_uint8_array(debug_file, (uint8_t*) instance_array[i].result2._limbs, num_bytes);

    fprintf(debug_file, "\n MYCUDA REDC: %d\n", i);
    fprint_uint8_array(debug_file, (uint8_t*) instance_array[i].winp._limbs, num_bytes);
    fprintf(debug_file, "\n MYCUDA REDC MSB: %d\n", i);
    fprint_uint8_array(debug_file, (uint8_t*)&(instance_array[i].winp._limbs[num_limbs_32]), num_bytes);
    fprintf(debug_file, "\n mul lo limbs: %d\n", i);
    fprint_uint8_array(debug_file, (uint8_t*) instance_array[i].mul_lo._limbs, num_bytes);
    fprintf(debug_file, "\n mul hi limbs: %d\n", i);
    fprint_uint8_array(debug_file, (uint8_t*) instance_array[i].mul_hi._limbs, num_bytes);

    // store the result.
    res_vector->emplace_back((uint8_t*)fresult);
  }
  free(num);
  free(modulus);
  free(instance_array);
  cudaFree(gpuInstances);
  return res_vector;
}
