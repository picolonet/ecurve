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

#include "p2_mul.cu"

using namespace libff;

const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;

// mnt4_q
  uint8_t mnt4_modulus[bytes_per_elem] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // mnt6_q
  uint8_t mnt6_modulus[bytes_per_elem] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

void fprint_fq(FILE* stream, Fq<mnt4753_pp> x) {
    int size = libff::mnt4753_q_limbs * sizeof(mp_size_t);
    uint8_t* array = (uint8_t*) x.mont_repr.data;
    for (int i = 0; i < size; i ++) {
        fprintf(stream, "%02x", array[i]);
    }
    fprintf(stream, "\n");
}

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

Fq<mnt4753_pp> to_fq(uint8_t* data) {
  Fq<mnt4753_pp> x;
  memcpy((void *) x.mont_repr.data, data, libff::mnt4753_q_limbs * sizeof(mp_size_t));
  return x;
}

void mul_play(std::vector<uint8_t*> x, std::vector<uint8_t*> y, FILE* debug_log) {
    mnt4753_pp::init_public_params();
    mnt6753_pp::init_public_params();

    std::vector<Fq<mnt4753_pp>> x0;
    std::vector<Fq<mnt4753_pp>> x1;

    int n = x.size();
    for (int i = 0; i < n; i++) {
       x0.emplace_back(to_fq(x[i]));
       x1.emplace_back(to_fq(y[i]));
       Fq<mnt4753_pp> out = x0[i] * x1[i];
       if (i < 2) {
       fprintf(debug_log, "\n X[%d]:", i);
       fprint_fq(debug_log, x0[i]);
       fprintf(debug_log, "\n Y[%d]:", i);
       fprint_fq(debug_log, x1[i]);
       fprintf(debug_log, "\n FQ[%d]:", i);
       fprint_fq(debug_log, out);
       }
    }
    // TODO: FIX ME FIX ME 
    // TODO: FIX ME FIX ME 
    // TODO: FIX ME FIX ME 
    // std::vector<uint8_t*>* result = compute_mont_mulcuda(x, y, mnt4_modulus, bytes_per_elem);
    std::vector<uint8_t*>* result = new std::vector<uint8_t*>();
    for (int i = 0; i < 2; i++) {
       fprintf(debug_log, "\n x[%d]:", i);
       fprint_uint8_array(debug_log, x[i], io_bytes_per_elem);
       fprintf(debug_log, "\n y[%d]:", i);
       fprint_uint8_array(debug_log, y[i], io_bytes_per_elem);
       fprintf(debug_log, "\n GPU[%d]:", i);
       fprint_uint8_array(debug_log, result->at(i), io_bytes_per_elem);
    }
}

void compute_gmp_inverse_32() {
   unsigned long l = 1;
   mpz_t n;
   mpz_init(n);
   mpz_set_str(n, "4294967296", 10);

   mpz_t m;
   mpz_init(m);
   unsigned long int m32;
   memcpy((void*)&m32, mnt4_modulus, sizeof(uint32_t));
   mpz_set_ui(m, m32);
   printf("\n setting m32 to %08X\n", m32);
   gmp_printf("\n setting n to %ZX\n", n);
   gmp_printf("\n setting n to %Zd\n", n);

   mpz_t rop;
   mpz_init(rop);
   mpz_invert(rop, m, n);
   gmp_printf (" mpz %Zd\n", rop);
   mpz_sub(rop, n, rop);
   gmp_printf (" mpz %Zd\n", rop);
   gmp_printf (" mpz %ZX\n", rop);
}


void compute_gmp_inverse() {
   unsigned long l = 1;
   mpz_t n;
   mpz_init(n);
   mpz_set_str(n, "18446744073709551616", 10);

   mpz_t m;
   mpz_init(m);
   unsigned long int m32;
   memcpy((void*)&m32, mnt4_modulus, sizeof(uint64_t));
   mpz_set_ui(m, m32);
   printf("\n setting m32 to %08X\n", m32);
   gmp_printf("\n setting n to %ZX\n", n);
   gmp_printf("\n setting n to %Zd\n", n);

   mpz_t rop;
   mpz_init(rop);
   mpz_invert(rop, m, n);
   gmp_printf (" mpz %Zd\n", rop);
   mpz_sub(rop, n, rop);
   gmp_printf (" mpz %Zd\n", rop);
   gmp_printf (" mpz %ZX\n", rop);
}

int main(int argc, char* argv[]) {
  printf("\nMain program. argc = %d \n", argc);
  
  // argv[2] = input_a;
  auto inputs = fopen(input_a, "r");
  auto debug_file = fopen("debug_log", "w");
  printf("\n Opening file %s for reading.\n", input_a);

  size_t n;
  clock_t start, end;
  double time_used = 0.0;
  double time_iter = 0.0;

  fprintf(debug_file, "\n mnt4 modulus:\n");
  fprint_uint8_array(debug_file, mnt4_modulus, bytes_per_elem); 

  printf("\n sieze of mplimb_t = %d, %d, %d", sizeof(mp_limb_t), sizeof(mp_size_t), libff::mnt4753_q_limbs);

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
    std::vector<uint8_t*>* result;
    // result = compute_add_cuda(x, y, mnt4_modulus, bytes_per_elem, debug_file);
    // result = compute_mul_const_cuda(x, y, mnt4_modulus, bytes_per_elem, debug_file);
    result = compute_mul_redc_cuda(x, y, mnt4_modulus, bytes_per_elem, MNT4_INV, debug_file);
    end = clock();

    time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
    time_used += time_iter;
    printf("\n GPU Round N, time = %5.4f ms.\n", time_iter); 
 
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
  fclose(debug_file);
}
