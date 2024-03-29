#include <cstdio>
#include <vector>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

using namespace libff;

void fprint_fq(FILE* stream, Fq<mnt4753_pp> x) {
    int size = libff::mnt4753_q_limbs * sizeof(mp_size_t);
    uint8_t* array = (uint8_t*) x.mont_repr.data;
    for (int i = 0; i < size; i ++) {
        fprintf(stream, "%02x", array[i]);
    }
    fprintf(stream, "\n");
    //printf("\nsize = %d * %d = %d\n", libff::mnt4753_q_limbs, sizeof(mp_size_t), size);
}

void fprint_fq6(FILE* stream, Fq<mnt6753_pp> x) {
    int size = libff::mnt6753_q_limbs * sizeof(mp_size_t);
    uint8_t* array = (uint8_t*) x.mont_repr.data;
    for (int i = 0; i < size; i ++) {
        fprintf(stream, "%02x", array[i]);
    }
    fprintf(stream, "\n");
    //printf("\nsize = %d * %d = %d\n", libff::mnt4753_q_limbs, sizeof(mp_size_t), size);
}

void write_mnt6_fq(FILE* output, Fq<mnt6753_pp> x) {
  fwrite((void *) x.mont_repr.data, libff::mnt6753_q_limbs * sizeof(mp_size_t), 1, output);
}

void print_fq(Fq<mnt4753_pp> x) {
    int size = libff::mnt4753_q_limbs * sizeof(mp_size_t);
    uint8_t* array = (uint8_t*) x.mont_repr.data;
    for (int i = 0; i < size; i ++) {
        printf("%02x", array[i]);
    }
    printf("\nsize = %d * %d = %d\n", libff::mnt4753_q_limbs, sizeof(mp_size_t), size);
}

void write_mnt4_fq(FILE* output, Fq<mnt4753_pp> x) {
  fwrite((void *) x.mont_repr.data, libff::mnt4753_q_limbs * sizeof(mp_size_t), 1, output);

  // printf("\n Output:\n");
  // fprint_fq(debug_log, x);
}

Fq<mnt4753_pp> read_mnt4_fq(FILE* input) {
  // bigint<mnt4753_q_limbs> n;
  Fq<mnt4753_pp> x;
  fread((void *) x.mont_repr.data, libff::mnt4753_q_limbs * sizeof(mp_size_t), 1, input);
  return x;
}

Fq<mnt6753_pp> read_mnt6_fq(FILE* input) {
  // bigint<mnt4753_q_limbs> n;
  Fq<mnt6753_pp> x;
  fread((void *) x.mont_repr.data, libff::mnt6753_q_limbs * sizeof(mp_size_t), 1, input);
  return x;
}

void play(Fq<mnt4753_pp> x, Fq<mnt4753_pp> y) {
  printf("\n MODULUS:");
  x.mod.print_hex();
  printf("\n\n\n");

  printf("\n PLAY INPUT X :");
  print_fq(x);
  printf("\n x.mont_repr:");
  x.mont_repr.print_hex();

  printf("\n PLAY INPUT Y :");
  print_fq(y);
  printf("\n y.mont_repr:");
  y.mont_repr.print_hex();

  Fq<mnt4753_pp> cx(131071);
  // Fq<mnt4753_pp> one(1);
  Fq<mnt4753_pp> sum = x.squared();
  printf("\n X * Y:");
  print_fq(sum);
  printf("\n x+x.mont_repr:");
  sum.mont_repr.print_hex();

  Fq<mnt4753_pp> mul = x * y;
  printf("\n X * Y:");
  print_fq(mul);
  printf("\n X*Y.mont_repr:");
  mul.mont_repr.print_hex();
}

// The actual code for doing Fq multiplication lives in libff/algebra/fields/fp.tcc
int main(int argc, char *argv[])
{
    // argv should be
    // { "main", "compute", inputs, outputs }

    mnt4753_pp::init_public_params();
    mnt6753_pp::init_public_params();

    size_t n;
    printf("\n size of mp_limb_t = %d", sizeof(mp_limb_t));

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");
    auto debug_log = fopen(argv[4], "w");

    while (true) {
      size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
      if (elts_read == 0) { break; }

      fprintf(debug_log, "\n\n NEW ROUND N = %d:", n);
      std::vector<Fq<mnt4753_pp>> x0;
      for (size_t i = 0; i < n; ++i) {
        x0.emplace_back(read_mnt4_fq(inputs));
        if (i < 5) {
          fprintf(debug_log, "\n Input X0[%d]:", i);
          fprint_fq(debug_log, x0.back());
        }
      }
      std::vector<Fq<mnt4753_pp>> x1;
      for (size_t i = 0; i < n; ++i) {
        x1.emplace_back(read_mnt4_fq(inputs));
        if (i < 5) {
          fprintf(debug_log, "\n Input X1[%d]:", i);
          fprint_fq(debug_log, x1.back());
        }
      }

      for (size_t i = 0; i < n; ++i) {
        Fq<mnt4753_pp> out = x0[i] * x1[i];
        write_mnt4_fq(outputs, out);
        if (i < 5) {
          fprintf(debug_log, "\n XOutput[%d]:", i);
          fprint_fq(debug_log, out);
        }
        // write_mnt4_fq(debug_log, outputs, x0[i] * x1[i]);
        //play(x0[i], x1[i]);
      }

      std::vector<Fq<mnt6753_pp>> y0;
      for (size_t i = 0; i < n; ++i) {
        y0.emplace_back(read_mnt6_fq(inputs));
        if (i < 5) {
          fprintf(debug_log, "\n Input Y0[%d]:", i);
          fprint_fq6(debug_log, y0.back());
        }
      }
      std::vector<Fq<mnt6753_pp>> y1;
      for (size_t i = 0; i < n; ++i) {
        y1.emplace_back(read_mnt6_fq(inputs));
        if (i < 5) {
          fprintf(debug_log, "\n Input Y1[%d]:", i);
          fprint_fq6(debug_log, y1.back());
        }
      }

      // printf("\n Input 0:\n");
      // print_fq(x0.front());

      // printf("\n Input 1:\n");
      // print_fq(x1.front());

      // printf("STARTING REAL COMPUTATION.");
      
      for (size_t i = 0; i < n; ++i) {
        Fq<mnt6753_pp> out = y0[i] * y1[i];
        write_mnt6_fq(outputs, out);
        if (i < 5) {
          fprintf(debug_log, "\n YOutput[%d]:", i);
          fprint_fq6(debug_log, out);
        }
        //write_mnt6_fq(debug_log, outputs, y0[i] * y1[i]);
      }
      //play(x0.front(), x1.front());
    }
    fclose(outputs);

    return 0;
}
