#include <cstdio>
#include <vector>

#include "ecurve.cu"

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

typedef struct G1Point{
  uint8_t* x;
  uint8_t* y;
  int size_bytes;
};

// This class represents two array of big integers.
// X and Y are points on an elliptic curve.
// We require x->size() == y->size().
class G1Array {

public:

  G1Array(std::vector<uint8_t*> *x, std::vector<uint8_t*> *y);

  std::vector<uint8_t*> *x, *y;

  // Reads an array of size num_elements into x and y.
  // This has assumptions on the file format.
  void read(FILE* inputs, int num_elements);

  void write(FILE* outputs);

  void setDebug(bool debug, FILE* debug_log);
 
  G1Point compute_cuda_sum();

  ~G1Array();

private:
  int io_bytes_per_elem_ = 96; 
  bool enable_debug_ = false;
  FILE* debug_log_ = NULL;

};

G1Array::G1Array(std::vector<uint8_t*> *x, std::vector<uint8_t*> *y) {
  this->x = x;
  this->y = y;
}

G1Array::~G1Array() {
  std::for_each(x->begin(), x->end(), delete_ptr());
  x->clear();

  std::for_each(y->begin(), y->end(), delete_ptr());
  y->clear();
}

void G1Array::read(FILE* inputs, int num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    x->emplace_back(my_read_mnt_fq(inputs, io_bytes_per_elem_));
    y->emplace_back(my_read_mnt_fq(inputs, io_bytes_per_elem_));
  }
}

void G1Array::write(FILE* outputs) {
  fwrite((void *) fq, io_bytes_per_elem_ * sizeof(uint8_t), 1, outputs);
}


void G1Array::setDebug(bool debug, FILE* debug_log) {
  enable_debug_ = debug;
  debug_log_ = debug_log;
}
  
G1Point G1Array::compute_cuda_sum() {
}

//
//G1Point* compute_cuda_sum(G1Point* a, G1Point* b) {
//}
