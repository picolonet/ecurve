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
  __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
      // use the mnt753 curve modulus
      fixnum mod_p(17);
      modnum mod = modnum(mod_p);

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

    uint8_t *input = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
        input[i] = 0;
    }

    uint8_t *input0 = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
        input0[i] = 0;
    }

    input[0] = 6;
    input0[0] = 11;

    fixnum_array *res, *in, *in0;
    in = fixnum_array::create(input, fn_bytes * nelts, fn_bytes);
    in0 = fixnum_array::create(input0, fn_bytes * nelts, fn_bytes);
    res = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res, in, in0);

    // TODO: input and output in montgomery form
    //   see 'modnum_monty_cios' vs 'modnum_monty_redc', idk the difference
    print_fixnum_array<fn_bytes, fixnum_array>(in, nelts);
    print_fixnum_array<fn_bytes, fixnum_array>(in0, nelts);
    print_fixnum_array<fn_bytes, fixnum_array>(res, nelts);

    //TODO to do stage 1 field arithmetic, instead of a map, do a reduce

    delete in;
    delete in0;
    delete res;
    delete[] input;
}

int main() {
    bench<8, u64_fixnum, convert_and_mul>(2);
    puts("");

    return 0;
}

