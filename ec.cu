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

void read_mnt4_g1(g1mfq_ti* out, FILE* input) {
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  auto inputs = fopen(argv[2], "r");
  auto outputs = fopen(argv[3], "w");
  size_t n;

  g1mfq_ti* g1m4_instances;

  while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }
    g1m4_instances = (g1_mfq_ti*) calloc(n, sizeof(g1mfq_ti));
    
    // Read G1 MNT4 input.
    for (size_t i = 0; i < n; ++i) { 

    free(g1m4_instances);
  }

  return 0;
}
