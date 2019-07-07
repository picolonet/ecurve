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

#include "myfq.h"
#include "utils.h"

const char* input_a = "/home/arunesh/github/snark-challenge/reference-01-field-arithmetic/inputs";


void test_fq_add(std::vector<uint8_t*> x, std::vector<uint8_t*> y, int num_bytes) {
  mnt4753_pp::init_public_params();
  mnt6753_pp::init_public_params();

  std::vector<Fq<mnt4753_pp>> x0;
  std::vector<Fq<mnt4753_pp>> x1;
  int n = x.size();
  for (int i = 0; i < n; i++) {
    x0.emplace_back(to_fq(x[i]));
    x1.emplace_back(to_fq(y[i]));
    Fq<mnt4753_pp> out = x0[i] * x1[i];
  }
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

  printf("\n Size of mplimb_t = %d, %d, %d", sizeof(mp_limb_t), sizeof(mp_size_t), libff::mnt4753_q_limbs);

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
    test_fq_add(x, y);
    end = clock();

    time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
    time_used += time_iter;
    //printf("\n GPU Round N, time = %5.4f ms.\n", time_iter); 
 
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

  fclose(inputs);
  fclose(debug_file);
}

