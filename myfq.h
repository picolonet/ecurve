#ifndef MYFQ_H 
#define  MYFQ_H


// List of Gpu params. BI generally stands for big integer.
struct MyGpuParams {

  static const int BI_BITS = 1024;

  static const int BI_BYTES = 128; 

  static const int BI_BITS_PER_LIMB = 64; 
   
  static const int BI_LIMBS = 16;

  static const int TPI = 16;  // Threads per instance, this has to match LIMBS per BigInt
};

// Fq really represents a biginteger of BI_LIMBS of type uint64_t. But since this is in
// CUDA, and gets parallely executed the class represents a single limb.
typedef struct MyFq {
    uint64_t &val;
} mfq_t;

// Class represents a big integer vector. But since it uses a GPU, all operations are
// defined on a single big integer which is of a fixed size.
// The basic data type is kept fixed at uint64_t.
typedef struct {
    mfq_t a0[MyGpuParams::BI_LIMBS];  
    mfq_t a1[MyGpuParams::BI_LIMBS];  
} mfq2_t;

typedef struct {
  mfq2_t A;
  mfq2_t B;
} mquad_t;


typedef struct {
  uint32_t lane;
  uint32_t sync_mask;
  uint32_t instance_number;
  uint32_t warp_number;
} thread_context_t;

__device__ void fq2_add(thread_context_t& tc, mfq2_t& a, mfq2_t& b);

__device__ void compute_context(thread_context_t& t);

#endif // MYFQ_H 
