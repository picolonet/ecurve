

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

     static m4g1* get(g1mfq_ti* instance, thread_context_t& tc);
     mfq_t x, y, z;
};

__global__
void fq_g1mfq_add_kernel(g1mfq_ti* instances, uint32_t instance_count) {
  // Create an array of m4g1 objects.
  // Implement functions.
  thread_context_t tc;
  compute_context(tc, instance_count);
  if (tc.instance_number >= instance_count) return;
  m4g1* my_instance = m4g1::get(instances[tc.instance_number], tc);
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
