#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <gmp.h>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

using namespace libff;
void fprint_fq(FILE* stream, Fq<mnt4753_pp> x);

void write_mnt4_fq(FILE* output, Fq<mnt4753_pp> x);

void write_mnt6_fq(FILE* output, Fq<mnt6753_pp> x);

void write_mnt4_fq2(FILE* output, Fqe<mnt4753_pp> x);

Fq<mnt4753_pp> read_mnt4_fq(FILE* input);

Fq<mnt6753_pp> read_mnt6_fq(FILE* input);

Fqe<mnt4753_pp> read_mnt4_fq2(FILE* input);

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

uint8_t* read_mnt_fq_2(FILE* inputs);

uint8_t* read_mnt_fq_2_gpu(FILE* inputs);

Fq<mnt4753_pp> to_fq(uint8_t* data);

bool check(uint8_t* a, uint8_t* b, int num);

void fprint_uint8_array(FILE* stream, uint8_t* array, int size);
