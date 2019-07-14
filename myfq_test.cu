#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>

#include <time.h>

#include <gmp.h>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "constants.h"

#include "myfq.cu"
#include "utils.cu"

const char* input_a = "/home/arunesh/github/snark-challenge/reference-01-field-arithmetic/inputs";

using namespace libff;

void test_fq_add(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file);
void test_fq_sub(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file);
void test_fq_mul_byconst(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file);
void test_fq_mont_mul(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file);
void test_fq_sub_newtest(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file);

struct fq_op { // Helper function to ease cleanup of container
    void operator () (std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) ;
};

struct add_fq_op : fq_op {
    void operator () (std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
        test_fq_add(x, y, num_bytes, debug_file);
    } 
};

struct sub_fq_op : fq_op {
    void operator () (std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
        test_fq_sub(x, y, num_bytes, debug_file);
    } 
};

struct sub_newtest: fq_op {
    void operator () (std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
        test_fq_sub_newtest(x, y, num_bytes, debug_file);
    } 
};

struct mul_by13_fq_op : fq_op {
    void operator () (std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
        test_fq_mul_byconst(x, y, num_bytes, debug_file);
    } 
};

struct mont_mul_fq_op : fq_op {
    void operator () (std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
        test_fq_mont_mul(x, y, num_bytes, debug_file);
    } 
};

uint8_t* read_mnt_fq_2_gpu(FILE* inputs) {
  uint8_t* buf; 
  cudaMallocManaged(&buf, bytes_per_elem , sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)buf, io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

template <typename fqop_temp>
void loadrun_fq_op(const char* input_file, const char* debug_filename) {
  // argv[2] = input_a;
  auto inputs = fopen(input_file, "r");
  auto debug_file = fopen(debug_filename, "w");
  printf("\n Opening file %s for reading.\n", input_a);

  size_t n;
  clock_t start, end;
  double time_used = 0.0;
  double time_iter = 0.0;

  fprintf(debug_file, "\n mnt4 modulus:\n");
  fprint_uint8_array(debug_file, mnt4_modulus, bytes_per_elem); 

  printf("\n Size of mplimb_t = %d, %d, %d", sizeof(mp_limb_t), sizeof(mp_size_t), libff::mnt4753_q_limbs);

  while(true) {
  
    size_t array_size = fread((void*) &n, sizeof(size_t), 1, inputs);
    if (array_size == 0) break;
    printf("\n Array size = %d\n", n);
    std::vector<uint8_t*> x;
    std::vector<uint8_t*> y;
    for (size_t i = 0; i < n; ++i) {
      uint8_t* ptr = read_mnt_fq_2(inputs);
      x.emplace_back(ptr);
    }
    for (size_t i = 0; i < n; ++i) {
      y.emplace_back(read_mnt_fq_2(inputs));
    }
    std::vector<uint8_t*> x6;
    std::vector<uint8_t*> y6;
    for (size_t i = 0; i < n; ++i) {
      x6.emplace_back(read_mnt_fq_2(inputs));
    }
    for (size_t i = 0; i < n; ++i) {
      y6.emplace_back(read_mnt_fq_2(inputs));
    }

    start = clock();
    fqop_temp()(x, y, bytes_per_elem, debug_file);
    end = clock();

    time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
    time_used += time_iter;
    printf("\n Full test function Round N, time = %5.4f ms.\n", time_iter); 
 
    std::for_each(x.begin(), x.end(), delete_ptr());
    x.clear();
    std::for_each(y.begin(), y.end(), delete_ptr());
    y.clear();
    std::for_each(x6.begin(), x6.end(), delete_ptr());
    x6.clear();
    std::for_each(y6.begin(), y6.end(), delete_ptr());
    y6.clear();
  } 
  fclose(inputs);
  fclose(debug_file);
}


void loadrun_fq_add(const char* input_file, const char* debug_filename) {
  // argv[2] = input_a;
  auto inputs = fopen(input_file, "r");
  auto debug_file = fopen(debug_filename, "w");
  printf("\n Opening file %s for reading.\n", input_a);

  size_t n;
  clock_t start, end;
  double time_used = 0.0;
  double time_iter = 0.0;

  fprintf(debug_file, "\n mnt4 modulus:\n");
  fprint_uint8_array(debug_file, mnt4_modulus, bytes_per_elem); 

  printf("\n Size of mplimb_t = %d, %d, %d", sizeof(mp_limb_t), sizeof(mp_size_t), libff::mnt4753_q_limbs);

  while(true) {
  
    size_t array_size = fread((void*) &n, sizeof(size_t), 1, inputs);
    if (array_size == 0) break;
    printf("\n Array size = %d\n", n);
    std::vector<uint8_t*> x;
    std::vector<uint8_t*> y;
    std::vector<uint8_t*> z;
    for (size_t i = 0; i < n; ++i) {
      uint8_t* ptr = read_mnt_fq_2(inputs);
      uint8_t* ptr2 = (uint8_t*)calloc(io_bytes_per_elem, sizeof(uint8_t));
      std::memcpy(ptr2, ptr, io_bytes_per_elem*sizeof(uint8_t));
      x.emplace_back(ptr);
      z.emplace_back(ptr2);
    }
    for (size_t i = 0; i < n; ++i) {
      y.emplace_back(read_mnt_fq_2(inputs));
    }
    std::vector<uint8_t*> x6;
    std::vector<uint8_t*> y6;
    for (size_t i = 0; i < n; ++i) {
      x6.emplace_back(read_mnt_fq_2(inputs));
    }
    for (size_t i = 0; i < n; ++i) {
      y6.emplace_back(read_mnt_fq_2(inputs));
    }

    int num_threads = io_bytes_per_elem / 8;

    start = clock();
    std::vector<uint8_t*>* result;
    test_fq_add(x, y, bytes_per_elem, debug_file);
    end = clock();

    time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
    time_used += time_iter;
    printf("\n Full test function Round N, time = %5.4f ms.\n", time_iter); 
 
    std::for_each(x.begin(), x.end(), delete_ptr());
    x.clear();
    std::for_each(y.begin(), y.end(), delete_ptr());
    y.clear();
    std::for_each(x6.begin(), x6.end(), delete_ptr());
    x6.clear();
    std::for_each(y6.begin(), y6.end(), delete_ptr());
    y6.clear();

  
  } 
  fclose(inputs);
  fclose(debug_file);
}

// no modulus MPN add that uses libgmp.
void mpn_mul_u64(uint8_t* a, uint64_t mul, int size_bytes) {
   mpn_mul_1((mp_limb_t *)a, (const mp_limb_t*) a, size_bytes / sizeof(mp_limb_t), mul);
}

// We test basic big int addition by a0 + a1 for a fq2 element.
void test_fq_mul_byconst(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
  mnt4753_pp::init_public_params();
  mnt6753_pp::init_public_params();

  std::vector<Fq<mnt4753_pp>> x0;
  std::vector<Fq<mnt4753_pp>> x1;
  cgbn_error_report_t *report;
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;

  int n = x.size();
  single_mfq_ti* gpuInstances;
  single_mfq_ti* localInstances;
  fprintf(debug_file, "\n size of fq2_t:%d", sizeof(single_mfq_ti));
  localInstances = (single_mfq_ti*) calloc(n, sizeof(single_mfq_ti));
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(single_mfq_ti)*n));
  load_mnt4_modulus();
  
  for (int i = 0; i < n; i++) {
      std::memcpy((void*)localInstances[i].x, (void*)x[i], num_bytes);
      localInstances[i].carry = 0;
      //std::memcpy((void*)localInstances[i].y, (void*)y[i], num_bytes);
  }
  
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, localInstances, sizeof(single_mfq_ti) * n, cudaMemcpyHostToDevice));
  //for (int i = 0; i < n; i++) {
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a0, x[i], num_bytes, cudaMemcpyHostToDevice));
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a1, y[i], num_bytes, cudaMemcpyHostToDevice));
  //}

  uint32_t num_blocks = (n + IPB-1)/IPB;
  clock_t start, end;
  double time_iter = 0.0;

  const uint64_t multiplier = 0x0FFFFFFFFFFFFFFFull;
  // const uint64_t multiplier = 0x0FFFull;

  n = 1; // CHANGE CHANGE
  start = clock();
  fq_mul_const_kernel<<<num_blocks, TPB>>>(gpuInstances, n, mnt4_modulus_device, multiplier);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  end = clock();
  time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
  fprintf(debug_file, "\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  printf("\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  NEW_CUDA_CHECK(cudaMemcpy(localInstances, gpuInstances, sizeof(single_mfq_ti) * n, cudaMemcpyDeviceToHost));
 
  Fq<mnt4753_pp> const13; 
  const13.set_ulong(multiplier);
  for (int i = 0; i < n; i++) {
    Fq<mnt4753_pp> out;
    x0.emplace_back(to_fq(x[i]));
    x1.emplace_back(to_fq(y[i]));
    out = x0[i] * const13;
    mpn_mul_u64(x[i], multiplier, num_bytes); 
    fprintf(debug_file, "\n REF MUL BY CONST:\n");
    fprint_uint8_array(debug_file, (uint8_t*)x[i], num_bytes); 
    fprintf(debug_file, "\n MY MUL BY CONST, carry = %d:\n", localInstances[i].carry);
    fprint_uint8_array(debug_file, (uint8_t*)localInstances[i].x, num_bytes); 
    if (check((uint8_t*) out.mont_repr.data, (uint8_t*)localInstances[i].x, io_bytes_per_elem)) {
      printf("\n TEST FAILED.");
      fprintf(debug_file, "\n TEST FAILED.");
    }
  }

  // free memory
  cudaFree(gpuInstances);
  free(localInstances);
}

// no modulus MPN add that uses libgmp.
void mpn_add(uint8_t* a, uint8_t* b, int size_bytes) {
   mpn_add_n((mp_limb_t *)a, (const mp_limb_t*) a, (const mp_limb_t*) b,
      size_bytes / sizeof(mp_limb_t));
}

// no modulus MPN sub A-B that uses libgmp. returns carry (borrow).
int mpn_sub(uint8_t* a, uint8_t* b, int size_bytes) {
   mpn_sub_n((mp_limb_t *)a, (const mp_limb_t*) a, (const mp_limb_t*) b,
      size_bytes / sizeof(mp_limb_t));
}

// We test basic big int addition by a0 + a1 for a fq2 element.
void test_fq_sub_newtest(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {

  mnt4753_pp::init_public_params();
  mnt6753_pp::init_public_params();

  std::vector<Fq<mnt4753_pp>> x0;
  std::vector<Fq<mnt4753_pp>> x1;
  cgbn_error_report_t *report;
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;

  int n = x.size();
  tuple_mfq_ti* gpuInstances;
  tuple_mfq_ti* localInstances;
  fprintf(debug_file, "\n size of fq2_t:%d", sizeof(tuple_mfq_ti));
  localInstances = (tuple_mfq_ti*) calloc(n, sizeof(tuple_mfq_ti));
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(tuple_mfq_ti)*n));
  load_mnt4_modulus();

  uint8_t* z = (uint8_t*) calloc(num_bytes, sizeof(uint8_t));
  for (int i = 0; i < n; i++) {
    mpn_add(y[i], x[i], num_bytes);
    std::memcpy((void*)localInstances[i].x, (void*)x[i], num_bytes);
    std::memcpy((void*)localInstances[i].y, (void*)y[i], num_bytes);
    localInstances[i].carry = 0;
  }

  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, localInstances, sizeof(tuple_mfq_ti) * n, cudaMemcpyHostToDevice));
  //for (int i = 0; i < n; i++) {
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a0, x[i], num_bytes, cudaMemcpyHostToDevice));
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a1, y[i], num_bytes, cudaMemcpyHostToDevice));
  //}

  uint32_t num_blocks = (n + IPB-1)/IPB;
  clock_t start, end;
  double time_iter = 0.0;

  start = clock();
  fq_sub_nomod_kernel<<<num_blocks, TPB>>>(gpuInstances, 1, mnt4_modulus_device);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  end = clock();
  time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
  fprintf(debug_file, "\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  printf("\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  NEW_CUDA_CHECK(cudaMemcpy(localInstances, gpuInstances, sizeof(tuple_mfq_ti) * n, cudaMemcpyDeviceToHost));
  
  for (int i = 0; i < 1; i++) {
    x0.emplace_back(to_fq(x[i]));
    x1.emplace_back(to_fq(y[i]));
    std::memcpy(z, x[i], num_bytes * sizeof(uint8_t));
    int carry = mpn_sub(z, y[i], num_bytes);
    fprintf(debug_file, "\n REF SUB, carry = %d:\n", carry);
    fprint_uint8_array(debug_file, (uint8_t*)z, num_bytes); 
    fprintf(debug_file, "\n MY SUB, carry = %d:\n", localInstances[i].carry);
    fprint_uint8_array(debug_file, (uint8_t*)localInstances[i].x, num_bytes); 
    if (check((uint8_t*) z, (uint8_t*)localInstances[i].x, io_bytes_per_elem)) {
      printf("\n TEST FAILED.");
      fprintf(debug_file, "\n TEST FAILED.");
    }
  }

  // free memory
  cudaFree(gpuInstances);
  free(localInstances);
  free(z);
}

// We test basic big int addition by a0 + a1 for a fq2 element.
void test_fq_sub(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
  mnt4753_pp::init_public_params();
  mnt6753_pp::init_public_params();

  std::vector<Fq<mnt4753_pp>> x0;
  std::vector<Fq<mnt4753_pp>> x1;
  cgbn_error_report_t *report;
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;

  int n = x.size();
  tuple_mfq_ti* gpuInstances;
  tuple_mfq_ti* localInstances;
  fprintf(debug_file, "\n size of fq2_t:%d", sizeof(tuple_mfq_ti));
  localInstances = (tuple_mfq_ti*) calloc(n, sizeof(tuple_mfq_ti));
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(tuple_mfq_ti)*n));
  load_mnt4_modulus();
  
  for (int i = 0; i < n; i++) {
      std::memcpy((void*)localInstances[i].x, (void*)x[i], num_bytes);
      std::memcpy((void*)localInstances[i].y, (void*)y[i], num_bytes);
  }
  
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, localInstances, sizeof(tuple_mfq_ti) * n, cudaMemcpyHostToDevice));
  //for (int i = 0; i < n; i++) {
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a0, x[i], num_bytes, cudaMemcpyHostToDevice));
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a1, y[i], num_bytes, cudaMemcpyHostToDevice));
  //}

  uint32_t num_blocks = (n + IPB-1)/IPB;
  clock_t start, end;
  double time_iter = 0.0;

  start = clock();
  fq_sub_kernel<<<num_blocks, TPB>>>(gpuInstances, n, mnt4_modulus_device);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  end = clock();
  time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
  fprintf(debug_file, "\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  printf("\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  NEW_CUDA_CHECK(cudaMemcpy(localInstances, gpuInstances, sizeof(tuple_mfq_ti) * n, cudaMemcpyDeviceToHost));
  
  for (int i = 0; i < n; i++) {
    x0.emplace_back(to_fq(x[i]));
    x1.emplace_back(to_fq(y[i]));
    Fq<mnt4753_pp> out = x0[i] - x1[i];
    fprintf(debug_file, "\n REF ADD:\n");
    fprint_fq(debug_file, out); 
    fprintf(debug_file, "\n MY ADD:\n");
    fprint_uint8_array(debug_file, (uint8_t*)localInstances[i].x, num_bytes); 
    if (check((uint8_t*) out.mont_repr.data, (uint8_t*)localInstances[i].x, io_bytes_per_elem)) {
      printf("\n TEST FAILED.");
      fprintf(debug_file, "\n TEST FAILED.");
    }
  }

  // free memory
  cudaFree(gpuInstances);
  free(localInstances);
}

// We test basic big int addition by a0 + a1 for a fq2 element.
void test_fq_mont_mul(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
  mnt4753_pp::init_public_params();
  mnt6753_pp::init_public_params();

  std::vector<Fq<mnt4753_pp>> x0;
  std::vector<Fq<mnt4753_pp>> x1;
  cgbn_error_report_t *report;
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;

  int n = x.size();
  triple_mfq_ti* gpuInstances;
  triple_mfq_ti* localInstances;
  fprintf(debug_file, "\n size of fq2_t:%d", sizeof(triple_mfq_ti));
  localInstances = (triple_mfq_ti*) calloc(n, sizeof(triple_mfq_ti));
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(triple_mfq_ti)*n));
  load_mnt4_modulus();
  
  for (int i = 0; i < n; i++) {
      std::memcpy((void*)localInstances[i].x, (void*)x[i], num_bytes);
      std::memcpy((void*)localInstances[i].y, (void*)y[i], num_bytes);
      std::memset((void*)localInstances[i].r, 0, num_bytes);
  }
  
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, localInstances, sizeof(triple_mfq_ti) * n, cudaMemcpyHostToDevice));
  //for (int i = 0; i < n; i++) {
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a0, x[i], num_bytes, cudaMemcpyHostToDevice));
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a1, y[i], num_bytes, cudaMemcpyHostToDevice));
  //}

  uint32_t num_blocks = (n + IPB-1)/IPB;
  clock_t start, end;
  double time_iter = 0.0;

  start = clock();
  fq_mont_mul_kernel<<<num_blocks, TPB>>>(gpuInstances, n, mnt4_modulus_device);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  end = clock();
  time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
  fprintf(debug_file, "\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  printf("\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  NEW_CUDA_CHECK(cudaMemcpy(localInstances, gpuInstances, sizeof(triple_mfq_ti) * n, cudaMemcpyDeviceToHost));
  
  for (int i = 0; i < n; i++) {
    x0.emplace_back(to_fq(x[i]));
    x1.emplace_back(to_fq(y[i]));
    Fq<mnt4753_pp> out = x0[i] * x1[i];
    fprintf(debug_file, "\n X_org:\n");
    fprint_fq(debug_file, x0[i]); 
    fprintf(debug_file, "\n X:\n");
    fprint_uint8_array(debug_file, (uint8_t*)localInstances[i].x, num_bytes); 
    fprintf(debug_file, "\n Y:\n");
    fprint_uint8_array(debug_file, (uint8_t*)localInstances[i].y, num_bytes); 
    fprintf(debug_file, "\n Y_orig:\n");
    fprint_uint8_array(debug_file, (uint8_t*)y[i], num_bytes); 
    fprintf(debug_file, "\n REF ADD:\n");
    fprint_fq(debug_file, out); 
    fprintf(debug_file, "\n MY ADD:\n");
    fprint_uint8_array(debug_file, (uint8_t*)localInstances[i].r, num_bytes); 
    if (check((uint8_t*) out.mont_repr.data, (uint8_t*)localInstances[i].r, io_bytes_per_elem)) {
      printf("\n TEST FAILED.");
      fprintf(debug_file, "\n TEST FAILED.");
    }
  }

  // free memory
  cudaFree(gpuInstances);
  free(localInstances);
}

// We test basic big int addition by a0 + a1 for a fq2 element.
void test_fq_add(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes, FILE* debug_file) {
  mnt4753_pp::init_public_params();
  mnt6753_pp::init_public_params();

  std::vector<Fq<mnt4753_pp>> x0;
  std::vector<Fq<mnt4753_pp>> x1;
  cgbn_error_report_t *report;
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;

  int n = x.size();
  tuple_mfq_ti* gpuInstances;
  tuple_mfq_ti* localInstances;
  fprintf(debug_file, "\n size of fq2_t:%d", sizeof(tuple_mfq_ti));
  localInstances = (tuple_mfq_ti*) calloc(n, sizeof(tuple_mfq_ti));
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(tuple_mfq_ti)*n));
  load_mnt4_modulus();
  
  for (int i = 0; i < n; i++) {
      std::memcpy((void*)localInstances[i].x, (void*)x[i], num_bytes);
      std::memcpy((void*)localInstances[i].y, (void*)y[i], num_bytes);
  }
  
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, localInstances, sizeof(tuple_mfq_ti) * n, cudaMemcpyHostToDevice));
  //for (int i = 0; i < n; i++) {
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a0, x[i], num_bytes, cudaMemcpyHostToDevice));
  //    NEW_CUDA_CHECK(cudaMemcpy(gpuInstances[i].a1, y[i], num_bytes, cudaMemcpyHostToDevice));
  //}

  uint32_t num_blocks = (n + IPB-1)/IPB;
  clock_t start, end;
  double time_iter = 0.0;

  start = clock();
  fq_add_kernel<<<num_blocks, TPB>>>(gpuInstances, n, mnt4_modulus_device);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  end = clock();
  time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
  fprintf(debug_file, "\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  printf("\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", n,
      time_iter, 1000.0*time_iter / (double)n); 
  NEW_CUDA_CHECK(cudaMemcpy(localInstances, gpuInstances, sizeof(tuple_mfq_ti) * n, cudaMemcpyDeviceToHost));
  
  for (int i = 0; i < n; i++) {
    x0.emplace_back(to_fq(x[i]));
    x1.emplace_back(to_fq(y[i]));
    Fq<mnt4753_pp> out = x0[i] + x1[i];
    fprintf(debug_file, "\n REF ADD:\n");
    fprint_fq(debug_file, out); 
    fprintf(debug_file, "\n MY ADD:\n");
    fprint_uint8_array(debug_file, (uint8_t*)localInstances[i].x, num_bytes); 
    if (check((uint8_t*) out.mont_repr.data, (uint8_t*)localInstances[i].x, io_bytes_per_elem)) {
      printf("\n TEST FAILED.");
      fprintf(debug_file, "\n TEST FAILED.");
    }
  }

  // free memory
  cudaFree(gpuInstances);
  free(localInstances);
}

int main(int argc, char* argv[]) {
  printf("\nMain program. argc = %d \n", argc);

  //loadrun_fq_add(input_a, "debug_log");
  //loadrun_fq_op<add_fq_op>(input_a, "debug_log");
  //loadrun_fq_op<sub_fq_op>(input_a, "debug_log");
  loadrun_fq_op<mul_by13_fq_op>(input_a, "debug_log");
  loadrun_fq_op<mont_mul_fq_op>(input_a, "debug_log");

  //loadrun_fq_op<sub_newtest>(input_a, "debug_log");
}

