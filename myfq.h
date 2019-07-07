#ifndef MYFQ_H 
#define  MYFQ_H


// List of Gpu params. BI generally stands for big integer.
struct MyGpuParams {

  const int BI_BITS = 1024;

  const int BI_BYTES = 128; 

  const int BI_BITS_PER_LIMB = 64; 
   
  const int BI_LIMBS = 16;

  const int TPI = 16;  // Threads per instance, this has to match LIMBS per BigInt
}

// Class represents a big integer vector. But since it uses a GPU, all operations are
// defined on a single big integer which is of a fixed size.
// The basic data type is kept fixed at uint64_t.
typedef struct {
    fq_t a0[BI_LIMBS];  
    fq_t a1[BI_LIMBS];  
} fq2_t;

typedef struct {
  fq2_t A;
  fq2_t B;
} quad_t


// Fq really represents a biginteger of BI_LIMBS of type uint64_t. But since this is in
// CUDA, and gets parallely executed the class represents a single limb.
typedef struct Fq {
    uint64_t &val;
} fq_t;

typedef struct {
  uint32_t lane;
  uint32_t sync_mask;
  uint32_t instance_number;
  uint32_t warp_number;
} thread_context_t;

__device__ void fq2_add(thread_context_t& tc, fq2_t& a, fq2_t& b);

__device__ void compute_context(thread_context_t& t);

#endif // MYFQ_H 
