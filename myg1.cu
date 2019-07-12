

#include "myfq.cu"

// Instance struct to pass the data to the GPU. Works for both MNT4 and MNT6.
typedef struct {
    mfq_t x[MyGpuParams::BI_LIMBS];  
    mfq_t y[MyGpuParams::BI_LIMBS];  
} g1mfq_ti;  // ti for instance, that is full array

typedef struct {
    mfq2_ti x, y;
} g1mfq2_ti; 

// G1 for the MNT4 curve.
class m4g1 {
  public:

     mfq_t x, y, z;
};
