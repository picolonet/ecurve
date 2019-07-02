#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>

#include <time.h>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

using namespace libff;

void write_mnt4_fq(FILE* output, Fq<mnt4753_pp> x) {
  fwrite((void *) x.mont_repr.data, libff::mnt4753_q_limbs * sizeof(mp_size_t), 1, output);
}

void write_mnt6_fq(FILE* output, Fq<mnt6753_pp> x) {
  fwrite((void *) x.mont_repr.data, libff::mnt6753_q_limbs * sizeof(mp_size_t), 1, output);
}

void write_mnt4_fq2(FILE* output, Fqe<mnt4753_pp> x) {
  write_mnt4_fq(output, x.c0);
  write_mnt4_fq(output, x.c1);
}

Fq<mnt4753_pp> read_mnt4_fq(FILE* input) {
  Fq<mnt4753_pp> x;
  fread((void *) x.mont_repr.data, libff::mnt4753_q_limbs * sizeof(mp_size_t), 1, input);
  return x;
}

Fq<mnt6753_pp> read_mnt6_fq(FILE* input) {
  Fq<mnt6753_pp> x;
  fread((void *) x.mont_repr.data, libff::mnt6753_q_limbs * sizeof(mp_size_t), 1, input);
  return x;
}

Fqe<mnt4753_pp> read_mnt4_fq2(FILE* input) {
  Fq<mnt4753_pp> c0 = read_mnt4_fq(input);
  Fq<mnt4753_pp> c1 = read_mnt4_fq(input);
  return Fqe<mnt4753_pp>(c0, c1);
}


struct delete_ptr { // Helper function to ease cleanup of container
    template <typename P>
    void operator () (P p) {
        delete p;
    }
};

struct delete_ptr_gpu { // Helper function to ease cleanup of container
    template <typename P>
    void operator () (P p) {
        cudaFree(p);
    }
};

const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;

const char* input_a = "/home/arunesh/github/snark-challenge/reference-01-field-arithmetic/inputs";

uint8_t* read_mnt_fq_2(FILE* inputs) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem, sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)buf, io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

uint8_t* read_mnt_fq_2_gpu(FILE* inputs) {
  uint8_t* buf; 
  cudaMallocManaged(&buf, bytes_per_elem , sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)buf, io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

__global__
void gpu_sum(uint64_t* a, uint64_t* b, int num) {
  int thread_id = threadIdx.x;
  printf("\n Gpu sum\n");
  //for (int i = 0; i < num; i ++) {
    uint64_t res = a[thread_id] + b[thread_id];
    printf("\nthread_id = %d, a[%08lX] + b[%08lX] = %08lX", thread_id, a[thread_id], b[thread_id], res);
    a[thread_id] = res; 
  //} 
}

void cpu_sum(uint64_t* a, uint64_t* b, int num) {
  uint64_t res;
  printf("\n Cpu sum\n");
  for (int i = 0; i < num; i ++) {
    res = a[i] + b[i];
    printf("\n i= %d a[%08lX] + b[%08lX] = %08lX", i, a[i], b[i], res);
    a[i] = res;
  } 
}

bool check(uint8_t* a, uint8_t* b, int num) {
  return memcmp(a, b, num * sizeof(uint8_t));
}

int main(int argc, char* argv[]) {
  printf("\nMain program. argc = %d \n", argc);
  
  // argv[2] = input_a;
  auto inputs = fopen(input_a, "r");
  printf("\n Opening file %s for reading.\n", input_a);

  size_t n;
  clock_t start, end;
  double time_used = 0.0;
  double time_iter = 0.0;

  while(true) {
  
    size_t array_size = fread((void*) &n, sizeof(size_t), 1, inputs);
    if (array_size == 0) break;
    printf("\n Array size = %d\n", n);
    std::vector<uint8_t*> x;
    std::vector<uint8_t*> y;
    std::vector<uint8_t*> z;
    for (size_t i = 0; i < n; ++i) {
      uint8_t* ptr = read_mnt_fq_2_gpu(inputs);
      uint8_t* ptr2 = (uint8_t*)calloc(io_bytes_per_elem, sizeof(uint8_t));
      std::memcpy(ptr2, ptr, io_bytes_per_elem*sizeof(uint8_t));
      x.emplace_back(ptr);
      z.emplace_back(ptr2);
    }
    for (size_t i = 0; i < n; ++i) {
      y.emplace_back(read_mnt_fq_2_gpu(inputs));
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
    for (size_t i = 0; i < 1; ++i) {
      gpu_sum<<< 1, num_threads >>>((uint64_t*) x[i], (uint64_t*) y[i], io_bytes_per_elem / 8);
    }
    cudaDeviceSynchronize();
    end = clock();

    time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
    time_used += time_iter;
    printf("\n GPU Round N, time = %5.4f ms.\n", time_iter); 
 
    start = clock();
    for (size_t i = 0; i < 1; ++i) {
       cpu_sum((uint64_t*)z[i], (uint64_t*)y[i], io_bytes_per_elem/8);
    }
    end = clock();

    time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n CPU Round N, time = %5.4f ms.\n", time_iter); 
    
    for (size_t i = 0; i < 1; ++i) {
      if (check(x[i], z[i], io_bytes_per_elem) != 0) {
        printf("\n Failed at %d.\n", i);
      }
    }
   
    std::for_each(x.begin(), x.end(), delete_ptr_gpu());
    x.clear();
    std::for_each(y.begin(), y.end(), delete_ptr_gpu());
    y.clear();
    std::for_each(x6.begin(), x6.end(), delete_ptr());
    x6.clear();
    std::for_each(y6.begin(), y6.end(), delete_ptr());
    y6.clear();

  
    break;
  } 
  
  printf("\n Total time = %5.4f ms.\n", time_used); 

  fclose(inputs);
}
