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

uint8_t* read_mnt_fq_2(FILE* inputs) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem, sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)buf, io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

bool check(uint8_t* a, uint8_t* b, int num) {
  return memcmp(a, b, num * sizeof(uint8_t));
}

void fprint_uint8_array(FILE* stream, uint8_t* array, int size) {
    for (int i = 0; i < size; i ++) {
        fprintf(stream, "%02x", array[i]);
    }
    fprintf(stream, "\n");
}

Fq<mnt4753_pp> to_fq(uint8_t* data) {
  Fq<mnt4753_pp> x;
  memcpy((void *) x.mont_repr.data, data, libff::mnt4753_q_limbs * sizeof(mp_size_t));
  return x;
}

