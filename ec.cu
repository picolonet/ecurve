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

#include "myg1.cu"
#include "utils.cu"


const int kNumIoBytes = io_bytes_per_elem;  // This should be 768.
const int kNumBytes = bytes_per_elem;  // This should be 1024.

int myread_mnt4_fq(mfq_t* out, FILE* input, int num_bytes, int num_io_bytes) {
  std::memset(out, 0, num_bytes * sizeof(uint8_t));
  return fread((void*)out, num_io_bytes*sizeof(uint8_t), 1, input);
}

int myread_mnt4_fq2(mfq2_ti* out, FILE* input, int num_bytes, int num_io_bytes) {
  myread_mnt4_fq(out->a0, input, num_bytes, num_io_bytes);
  return myread_mnt4_fq(out->a1, input, num_bytes, num_io_bytes);
}

void myread_mnt4_g1(g1mfq_ti* out, FILE* input, int num_bytes, int num_io_bytes) {
  myread_mnt4_fq(out->x, input, num_bytes, num_io_bytes);
  myread_mnt4_fq(out->y, input, num_bytes, num_io_bytes);
}

void myread_mnt4_g1fq(g1mfq_ti* out, FILE* input, int num_bytes, int num_io_bytes) {
  myread_mnt4_fq(out->x, input, num_bytes, num_io_bytes);
  myread_mnt4_fq(out->y, input, num_bytes, num_io_bytes);
}

void myread_mnt4_g1fq2(g1mfq2_ti* out, FILE* input, int num_bytes, int num_io_bytes) {
  myread_mnt4_fq2(&out->x, input, num_bytes, num_io_bytes);
  myread_mnt4_fq2(&out->y, input, num_bytes, num_io_bytes);
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  auto input = fopen(argv[2], "r");
  auto output = fopen(argv[3], "w");
  size_t n;

  g1mfq_ti* g1m4_instances;
  g1mfq2_ti* g1m4fq2_instances;

  while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, input);
    if (elts_read == 0) { break; }
    g1m4_instances = (g1mfq_ti*) calloc(n, sizeof(g1mfq_ti));
    g1m4fq2_instances = (g1mfq2_ti*) calloc(n, sizeof(g1mfq2_ti));

    // Read G1 MNT4 input.
    for (size_t i = 0; i < n; ++i) { 
      myread_mnt4_g1fq(g1m4_instances + i, input, kNumBytes, kNumIoBytes);
    }

    for (size_t i = 0; i < n; ++i) { 
      myread_mnt4_g1fq2(g1m4fq2_instances + i, input, kNumBytes, kNumIoBytes);
    }

    //  Optional vector based access.
    // std::vector<g1mfq_ti> g1m4_instances(g1m4_instances, g1m4_instances + (n-1));
    free(g1m4_instances);
  }
  fclose(input);

  return 0;
}
