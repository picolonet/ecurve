#include "cubex.cu"


struct delete_ptr {
  template <typename P>
    void operator () (P ptr) {
       delete ptr;
    }
};

class G1Array {

public:

  std::vector<uint8_t*> *x, *y;

  ~G1Array();
};

G1Array::~G1Array() {
  std::for_each(x->begin(), x->end(), delete_ptr());
  x->clear();

  std::for_each(y->begin(), y->end(), delete_ptr());
  y->clear();
}

