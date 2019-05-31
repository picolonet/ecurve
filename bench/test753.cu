#include <cstdio>
#include <cstring>
#include <cassert>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

using namespace std;
using namespace cuFIXNUM;

template< typename fixnum >
struct convert_and_mul {
  typedef modnum_monty_redc<fixnum> modnum;
  __device__ void operator()(fixnum &r, fixnum a, fixnum b, fixnum my_mod) {
      modnum mod = modnum(my_mod);

      fixnum sm;
      fixnum am;
      fixnum bm;

      mod.to_modnum(am, a);
      mod.to_modnum(bm, b);

      mod.mul(sm, am, bm);

      fixnum s;
      mod.from_modnum(s, sm);

      r = s;
  }
};

template< int fn_bytes, typename fixnum_array >
void print_fixnum_array(fixnum_array* res, int nelts) {

    int lrl = fn_bytes*nelts;
    uint8_t local_results[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);

    for (int i = 0; i < lrl; i++) {
      printf("%i ", local_results[i]);
    }
    printf("\n");
}


template< int fn_bytes, typename word_fixnum, template <typename> class Func >
void bench(int nelts) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    uint8_t input_m[128] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    uint8_t *input = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
        input[i] = input_m[i];
    }

    uint8_t *input0 = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
        input0[i] = input_m[i];
    }

    input[0] = 0;
    input0[0] = 0;

    // TODO reuse modulus as a constant instead of passing in nelts times
    fixnum_array *res, *in, *in0, *inM;
    in = fixnum_array::create(input, fn_bytes * nelts, fn_bytes);
    in0 = fixnum_array::create(input0, fn_bytes * nelts, fn_bytes);
    inM = fixnum_array::create(input_m, fn_bytes * nelts, fn_bytes);
    res = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res, in, in0, inM);

    // TODO: input and output in montgomery form
    //   see 'modnum_monty_cios' vs 'modnum_monty_redc', idk the difference
    print_fixnum_array<fn_bytes, fixnum_array>(in, nelts);
    print_fixnum_array<fn_bytes, fixnum_array>(in0, nelts);
    print_fixnum_array<fn_bytes, fixnum_array>(res, nelts);

    //TODO to do stage 1 field arithmetic, instead of a map, do a reduce

    delete in;
    delete in0;
    delete inM;
    delete res;
    delete[] input;
}

int main() {
    bench<128, u64_fixnum, convert_and_mul>(1);
    puts("");

    return 0;
}

