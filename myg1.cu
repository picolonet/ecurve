

#include "myfq.cu"

__constant__ mfq_t g1_mnt4_coeff_a[16];
__constant__ mfq_t g1_mnt4_coeff_b[16];

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

__device__
     static m4g1* get(g1mfq_ti* instance, thread_context_t& tc);

__device__
     void plusEquals(m4g1& x);

__device__
     void dbl();
   
__device__
     bool is_zero();

     mfq_t x, y, z;
     thread_context_t& tc;
};
__device__
bool m4g1::is_zero() {
  uint32_t x_flag, z_flag;
  x_flag = __ballot_sync(tc.sync_mask, x==0);
  z_flag = __ballot_sync(tc.sync_mask, z==0);
  return (x_flag & tc.sync_mask & z_flag) == tc.sync_mask;
}

void load_g1_mnt4_coeffs() {
  cudaMemcpyToSymbol(g1_mnt4_coeff_a, mnt4_g1_coeff_a, bytes_per_elem, 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(g1_mnt4_coeff_b, mnt4_g1_coeff_b, bytes_per_elem, 0, cudaMemcpyHostToDevice);
}

void load_constants() {
  load_mnt4_modulus();
  load_g1_mnt4_coeffs();
}

// TODO: Handle is_zero and is_one conditions.
__device__
m4g1* m4g1::get(g1mfq_ti* instance, thread_context_t& tc) {
  // TODO build
  return NULL;
}
     
__device__
void m4g1::plusEquals(m4g1& x) {
  // TODO build
}

__device__
void m4g1::dbl() {

  // TODO build
  if (is_zero()) return;
  const myfq_t XX = myfq_square(x);
  const myfq_t ZZ = myfq_square(z);
  myfq_t aZZ;  
  mont_mul_64_lane(tc, aZZ, g1_mnt4_coeff_a[tc.lane], ZZ, mnt4_modulus_device, MNT4_INV, 12);

//  mfq_t XX = my
// NOTE: does not handle O and pts of order 2,4
        // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-bl

//        const mnt4753_Fq XX   = (this->X_).squared();                   // XX  = X1^2
//        const mnt4753_Fq ZZ   = (this->Z_).squared();                   // ZZ  = Z1^2
//        const mnt4753_Fq w    = mnt4753_G1::coeff_a * ZZ + (XX + XX + XX); // w   = a*ZZ + 3*XX
//        const mnt4753_Fq Y1Z1 = (this->Y_) * (this->Z_);
//        const mnt4753_Fq s    = Y1Z1 + Y1Z1;                            // s   = 2*Y1*Z1
//        const mnt4753_Fq ss   = s.squared();                            // ss  = s^2
//        const mnt4753_Fq sss  = s * ss;                                 // sss = s*ss
//        const mnt4753_Fq R    = (this->Y_) * s;                         // R   = Y1*s
//        const mnt4753_Fq RR   = R.squared();                            // RR  = R^2
//        const mnt4753_Fq B    = ((this->X_)+R).squared()-XX-RR;         // B   = (X1+R)^2 - XX - RR
//        const mnt4753_Fq h    = w.squared() - (B+B);                    // h   = w^2 - 2*B
//        const mnt4753_Fq X3   = h * s;                                  // X3  = h*s
//        const mnt4753_Fq Y3   = w * (B-h)-(RR+RR);                      // Y3  = w*(B-h) - 2*RR
//        const mnt4753_Fq Z3   = sss;                                    // Z3  = sss
//
//        return mnt4753_G1(X3, Y3, Z3);

}

__global__
void fq_g1mfq_add_kernel(g1mfq_ti* instances, uint32_t instance_count) {
  // Create an array of m4g1 objects.
  // Implement functions.
  thread_context_t tc;
  compute_context(tc, instance_count);
  if (tc.instance_number >= instance_count) return;
  m4g1* my_instance = m4g1::get(&instances[tc.instance_number], tc);
}

__global__
void fq_g1mfq2_add_kernel(g1mfq2_ti* instances, uint32_t instance_count) {
  // Create an array of m4g1 objects
}

void compute_g1fq_sum(g1mfq_ti* instances, uint32_t instance_count, FILE* debug_file) {
  cgbn_error_report_t *report;
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;

  g1mfq_ti* gpuInstances;
  fprintf(debug_file, "\n size of g1mfq_ti:%d", sizeof(g1mfq_ti));
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(g1mfq_ti) * instance_count));
  load_mnt4_modulus();
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(g1mfq_ti) * instance_count,
     cudaMemcpyHostToDevice));
  uint32_t num_blocks = (instance_count + IPB-1)/IPB;
  clock_t start, end;
  double time_iter = 0.0;

  start = clock();
  fq_g1mfq_add_kernel<<<num_blocks, TPB>>>(gpuInstances, instance_count);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  end = clock();

  time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
  fprintf(debug_file, "\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", instance_count,
      time_iter, 1000.0*time_iter / (double)instance_count); 
  printf("\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", instance_count,
      time_iter, 1000.0*time_iter / (double)instance_count); 
  NEW_CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(g1mfq_ti) * instance_count, cudaMemcpyDeviceToHost));
}

void compute_g1fq2_sum(g1mfq2_ti* instances, uint32_t instance_count, FILE* debug_file) {
  cgbn_error_report_t *report;
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));

  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;

  g1mfq2_ti* gpuInstances;
  fprintf(debug_file, "\n size of g1mfq2_ti:%d", sizeof(g1mfq2_ti));
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(g1mfq2_ti) * instance_count));
  load_mnt4_modulus();
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(g1mfq2_ti) * instance_count,
     cudaMemcpyHostToDevice));
  uint32_t num_blocks = (instance_count + IPB-1)/IPB;
  clock_t start, end;
  double time_iter = 0.0;

  start = clock();
  fq_g1mfq2_add_kernel<<<num_blocks, TPB>>>(gpuInstances, instance_count);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  end = clock();

  time_iter = ((double) end-start) * 1000.0 / CLOCKS_PER_SEC;
  fprintf(debug_file, "\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", instance_count,
      time_iter, 1000.0*time_iter / (double)instance_count); 
  printf("\n num_elements = %d, compute ony latency = %8.7f ms, per element = %8.7f microseconds.\n", instance_count,
      time_iter, 1000.0*time_iter / (double)instance_count); 
  NEW_CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(g1mfq2_ti) * instance_count, cudaMemcpyDeviceToHost));
}
