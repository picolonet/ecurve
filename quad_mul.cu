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

#define TPB 128    // the number of threads per block to launch (must be divisible by 32

typedef struct {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> y;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> mul_lo;
  cgbn_mem_t<BITS> mul_hi;
} my_instance_t;


typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, 768> env1024_t;

const uint64_t MNT4_INV = 0xf2044cfbe45e7fff;
const uint64_t MNT6_INV = 0xc90776e23fffffff;


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

__global__ void my_kernel(my_instance_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context;                                 // create a CGBN context
  env1024_t         bn1024_env(bn_context);                     // construct a bn environment for 1024 bit math
  env1024_t::cgbn_t a, b, m;                      // three 1024-bit values (spread across a warp)
  env1024_t::cgbn_wide_t mul_wide;
  // uint32_t np0;
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
  
  cgbn_load(bn1024_env, a, &(problem_instances[my_instance]).x);
  cgbn_load(bn1024_env, b, &(problem_instances[my_instance]).y);
  cgbn_load(bn1024_env, m, &(problem_instances[my_instance]).m);

  // np0 = -cgbn_binary_inverse_ui32(bn1024_env, cgbn_get_ui32(bn1024_env, m));

  cgbn_mul_wide(bn1024_env, mul_wide, a, b);

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

std::vector<uint8_t*>* compute_newcuda(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t* input_m_base, int num_bytes, uint64_t inv) {
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

  my_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(my_instance_t)*num_elements, cudaMemcpyDeviceToHost));


  int num_limbs = num_bytes / 8;
  printf("\n Setting num 64 limbs = %d", num_limbs);
  mp_limb_t* num = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs * 2);
  mp_limb_t* modulus = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);
  std::memcpy((void*) modulus, (const void*) instance_array->m._limbs, num_bytes);

  printf("\n Dumping modulus:");
  gmp_printf("%Nx\n", modulus, num_limbs); 

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
    // Reduce
    std::memcpy((void*)num, (const void*)instance_array[i].mul_lo._limbs, num_bytes);
    std::memcpy((void*) (num + num_limbs), (const void*)instance_array[i].mul_hi._limbs, num_bytes);
    mp_limb_t* fresult = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);
 
    // printf("\n Dumping 64 byte limb wide num [%d]:", i);
    // gmp_printf("%Nx\n", num, num_limbs * 2); 

    reduce_wide(fresult, num, modulus, inv, num_limbs);

    // store the result.
    res_vector->emplace_back((uint8_t*)fresult);
  }
  free(num);
  free(modulus);
  free(instance_array);
  cudaFree(gpuInstances);
  return res_vector;
}


std::pair<std::vector<uint8_t*>, std::vector<uint8_t*> > 
compute_quadex_cuda(std::vector<uint8_t*> x0_a0,
                    std::vector<uint8_t*> x0_a1,
                    std::vector<uint8_t*> y0_a0,
                    std::vector<uint8_t*> y0_a1,
                    uint8_t* input_m_base, int num_bytes, uint64_t inv) {
  int num_elements = x0_a0.size();
  std::pair<std::vector<uint8_t*>, std::vector<uint8_t*> > res;
  return res;
}
