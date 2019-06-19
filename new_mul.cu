#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <cassert>
#include "cgbn/cgbn.h"
#include "utility/support.h"


#define TPI 32
#define BITS 768 

#define TPB 32    // the number of threads per block to launch (must be divisible by 32

typedef struct {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> y;
  cgbn_mem_t<BITS> l;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> result;
  cgbn_mem_t<BITS> result2;
  cgbn_mem_t<BITS> mul_lo;
  cgbn_mem_t<BITS> mul_hi;
} instance_t;

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, 768> env1024_t;


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

__device__
 void store_np0(env1024_t::cgbn_t& l, uint32_t np0) {
  #if defined(__CUDA_ARCH__)
  #warning "including limbs code"
   l._limbs[10] = np0;
   l._limbs[11] = 0xe45e7fffu;
   printf("one %x, np-0 = %x\n", l._limbs[10], l._limbs[11]);
  #endif
}
__global__ void add_kernel(instance_t *problem_instances, uint32_t instance_count, int add_pow_count) {
  context_t         bn_context;                                 // create a CGBN context
  env1024_t         bn1024_env(bn_context);                     // construct a bn environment for 1024 bit math
  env1024_t::cgbn_t a, b, mul_r, add_r, add_r1, add_r2, acc_r, acc_r1, acc_r2, m, l;                      // three 1024-bit values (spread across a warp)
  env1024_t::cgbn_t mul_r2;
  env1024_t::cgbn_wide_t mul_wide;
  uint32_t np0;
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
  
  cgbn_load(bn1024_env, a, &(problem_instances[my_instance]).x);
  cgbn_load(bn1024_env, b, &(problem_instances[my_instance]).y);
  cgbn_load(bn1024_env, m, &(problem_instances[my_instance]).m);
  cgbn_load(bn1024_env, l, &(problem_instances[my_instance]).l);
  
  // cgbn_add(bn1024_env, r, a, b);
  np0 = -cgbn_binary_inverse_ui32(bn1024_env, cgbn_get_ui32(bn1024_env, m));

  #if defined(__CUDA_ARCH__)
  l._limbs[12] = np0;
  #endif
  np0=cgbn_bn2mont(bn1024_env, l, l, m);
  // cgbn_bn2mont(bn1024_env, b, b, m);
  cgbn_mont_mul(bn1024_env, mul_r, a, b, m, np0);
  cgbn_mul_wide(bn1024_env, mul_wide, a, b);
  if (cgbn_compare(bn1024_env, mul_r, m) >= 0) {
       cgbn_sub(bn1024_env, add_r, mul_r, m);
       cgbn_set(bn1024_env, mul_r, add_r); 
  }
  
  cgbn_set(bn1024_env, add_r, a); 
  cgbn_set(bn1024_env, acc_r, a); 
  for (int i = 0; i < add_pow_count; i ++) {
    cgbn_add(bn1024_env, add_r1, add_r, add_r);
    if (cgbn_compare(bn1024_env, add_r1, m) >= 0) {
       cgbn_sub(bn1024_env, add_r2, add_r1, m);
       cgbn_set(bn1024_env, add_r, add_r2); 
    } else {
       cgbn_set(bn1024_env, add_r, add_r1); 
    }

    cgbn_add(bn1024_env, acc_r1, acc_r, add_r);
    if (cgbn_compare(bn1024_env, acc_r1, m) >= 0) {
       cgbn_sub(bn1024_env, acc_r2, acc_r1, m);
       cgbn_set(bn1024_env, acc_r, acc_r2); 
    } else {
       cgbn_set(bn1024_env, acc_r, acc_r1); 
    }
  }
  cgbn_store(bn1024_env, &(problem_instances[my_instance].result), acc_r);

  //cgbn_mont2bn(bn1024_env, r, r, m, np0);
  // int use_r2 = cgbn_sub(bn1024_env, add_r2, add_r1, m);
  
  // if (use_r2 == 0) {
  // } else {
  //  cgbn_store(bn1024_env, &(problem_instances[my_instance].result), add_r1);
  // }
  cgbn_mont_reduce_wide(bn1024_env, mul_r2, mul_wide, m, 0xe45e7fffu);
  cgbn_store(bn1024_env, &(problem_instances[my_instance].result2), mul_r2);
  cgbn_store(bn1024_env, &(problem_instances[my_instance].mul_lo), mul_wide._low);
  cgbn_store(bn1024_env, &(problem_instances[my_instance].mul_hi), mul_wide._high);
}

void set_literal(cgbn_mem_t<BITS>& h, uint32_t literal, int num) {
  for (int i = 1; i < num; i ++ ) {
     h._limbs[i] = 0;
  }
  h._limbs[0] = literal;
}

void set_literal_limbs(cgbn_mem_t<BITS>& h, uint32_t literal, int num, int size) {
  for (int i = 0; i < num; i ++ ) {
     h._limbs[i] = literal;
  }
  for (int i = num; i < size; i ++ ) {
     h._limbs[i] = 0;
  }
}

void print_uint8_array(uint8_t* array, int size) {
    for (int i = 0; i < size; i ++) {
        printf("%02x", array[i]);
    }
    printf("\n");
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

  set_literal_limbs(instance_array->l, (uint32_t)0xFFFFFFFF, 2, num_bytes/4);
  printf("\n L:");
  print_uint8_array((uint8_t*) instance_array->l._limbs, num_bytes);

  printf("Copying instances to the GPU ...\n");
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*count));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(instance_t)*count, cudaMemcpyHostToDevice));

  add_kernel<<<1, 32>>>(gpuInstances, 1, 63);
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

  printf("Printing mont-mul result:");
  print_uint8_array((uint8_t*) instance_array->result2._limbs, num_bytes);
  printf("Printing mont-mul HI result:");
  print_uint8_array((uint8_t*) instance_array->mul_hi._limbs, num_bytes);
  printf("Printing mont-mul LOW result:");
  print_uint8_array((uint8_t*) instance_array->mul_lo._limbs, num_bytes);
  printf("Done. returning ...\n");

  int num_limbs = num_bytes / 8;
  printf("\n Setting num 64 limbs = %d", num_limbs);
  mp_limb_t* num = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs * 2);
  std::memcpy((void*)num, (const void*)instance_array->mul_lo._limbs, num_bytes);
  std::memcpy((void*) (num + num_limbs), (const void*)instance_array->mul_lo._limbs, num_bytes);
 
  printf("\n Dumping 64 byte limb wide num:");
  gmp_printf("%Nx\n", num, num_limbs * 2); 

  mp_limb_t* modulus = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);
  std::memcpy((void*) modulus, (const void*) instance_array->m._limbs, num_bytes);

  mp_limb_t* fresult = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);

  free(num);
  free(modulus);
  //free(fresult);

  printf("\n Dumping 64 byte modulus:");
  gmp_printf("%Nx\n", m, num_limbs); 
  return result;
}
