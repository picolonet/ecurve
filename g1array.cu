#include <cstdio>
#include <vector>

#include "cubex.cu"

uint8_t* my_read_mnt_fq(FILE* inputs, int bytes_per_elem) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem, sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)buf, bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

struct delete_ptr {
  template <typename P>
    void operator () (P ptr) {
       delete ptr;
    }
};

// This class represents two array of big integers.
// X and Y are points on an elliptic curve.
// We require x->size() == y->size().
class G1Array {

public:

  std::vector<uint8_t*> *x, *y;

  // Reads an array of size num_elements into x and y.
  // This has assumptions on the file format.
  void read(FILE* inputs, int num_elements);

  void write(FILE* outputs);

  ~G1Array();

private:
  int io_bytes_per_elem = 96; 

};

G1Array::~G1Array() {
  std::for_each(x->begin(), x->end(), delete_ptr());
  x->clear();

  std::for_each(y->begin(), y->end(), delete_ptr());
  y->clear();
}

void G1Array::read(FILE* inputs, int num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    x->emplace_back(my_read_mnt_fq(inputs, io_bytes_per_elem));
    y->emplace_back(my_read_mnt_fq(inputs, io_bytes_per_elem));
  }
}

void G1Array::write(FILE* outputs) {
  fwrite((void *) fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}
