#include <cstdio>
#include <vector>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

using namespace libff;

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
  // print_fq(x);
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

  // Fq<mnt4753_pp> one(1);
  Fq<mnt4753_pp> sum = y * x;
  printf("\n X + Y:");
  print_fq(sum);
  printf("\n x+y.mont_repr:");
  sum.mont_repr.print_hex();
}

// The actual code for doing Fq multiplication lives in libff/algebra/fields/fp.tcc
int main(int argc, char *argv[])
{
    // argv should be
    // { "main", "compute", inputs, outputs }

    mnt4753_pp::init_public_params();
    mnt6753_pp::init_public_params();

    size_t n;

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");

    while (true) {
      size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
      if (elts_read == 0) { break; }

      std::vector<Fq<mnt4753_pp>> x0;
      for (size_t i = 0; i < n; ++i) {
        x0.emplace_back(read_mnt4_fq(inputs));
      }
      std::vector<Fq<mnt4753_pp>> x1;
      for (size_t i = 0; i < n; ++i) {
        x1.emplace_back(read_mnt4_fq(inputs));
      }

      std::vector<Fq<mnt6753_pp>> y0;
      for (size_t i = 0; i < n; ++i) {
        y0.emplace_back(read_mnt6_fq(inputs));
      }
      std::vector<Fq<mnt6753_pp>> y1;
      for (size_t i = 0; i < n; ++i) {
        y1.emplace_back(read_mnt6_fq(inputs));
      }

      // printf("\n Input 0:\n");
      // print_fq(x0.front());

      // printf("\n Input 1:\n");
      // print_fq(x1.front());

      for (size_t i = 0; i < n; ++i) {
        write_mnt4_fq(outputs, x0[i] * x1[i]);
      }

      for (size_t i = 0; i < n; ++i) {
        write_mnt6_fq(outputs, y0[i] * y1[i]);
      }
      play(x0.front(), x1.front());
    }
    fclose(outputs);

    return 0;
}
