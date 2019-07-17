

#include "myfq.cu"

__constant__ mfq_t g1_mnt4_coeff_a[16];
__constant__ mfq_t g1_mnt4_coeff_b[16];

// Instance struct to pass the data to the GPU. Works for both MNT4 and MNT6.
typedef struct {
    mfq_t x[MyGpuParams::BI_LIMBS];  
    mfq_t y[MyGpuParams::BI_LIMBS];  
    mfq_t z[MyGpuParams::BI_LIMBS];  
} g1mfq_ti;  // ti for instance, that is full array

typedef struct {
    mfq2_ti x, y;
} g1mfq2_ti; 

// G1 for the MNT4 curve.
class m4g1 {
  public:
__device__
     m4g1(mfq_t& x_, mfq_t& y_, mfq_t& z_) : x(x_), y(y_), z(z_) {}
__device__
     m4g1(m4g1 other) : x(x_), y(y_), z(z_) {}

__device__
     void plusEquals(thread_context_t& tc, m4g1& x);

__device__
     void dbl(thread_context_t& tc);
   
__device__
     bool is_zero(thread_context_t& tc);

__device__
     void to_affine(thread_context_t& tc);

__device__
     static m4g1 zero();

     mfq_t& x;
     mfq_t& y;
     mfq_t& z;
};

__device__
static m4g1 m4g1::zero() {
  return m4g1(0, mfq_one(), 0);
}

__device__
bool m4g1::is_zero(thread_context_t& tc) {
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

__device__
void m4g1::to_affine(thread_context_t& tc) {
}

// TODO: Handle is_zero and is_one conditions.
//__device__
//m4g1* m4g1::get(g1mfq_ti* instance, thread_context_t& tc) {
//  // TODO build
//  return NULL;
//}
     
__device__
void m4g1::plusEquals(thread_context_t& tc, m4g1& other) {
  // TODO build
  if (is_zero(tc)) {
    return other;
  }

  if (other.is_zero(tc)) {
    return *this;
  }
 
  // Algo from: libff/algebra/curves/mnt753/mnt4753/mnt4753_g1.cpp#L164 
  
  // const mnt4753_Fq X1Z2 = (this->X_) * (other.Z_);        // X1Z2 = X1*Z2
  mfq_t X1Z2;
  mont_mul_64_lane(tc, X1Z2, x, other.z, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq X2Z1 = (this->Z_) * (other.X_);        // X2Z1 = X2*Z1
  mfq_t X2Z1;
  mont_mul_64_lane(tc, X2Z1, z, other.x, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq Y1Z2 = (this->Y_) * (other.Z_);        // Y1Z2 = Y1*Z2
  mfq_t Y1Z2;
  mont_mul_64_lane(tc, Y1Z2, y, other.z, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq Y2Z1 = (this->Z_) * (other.Y_);        // Y2Z1 = Y2*Z1
  mfq_t Y2Z1;
  mont_mul_64_lane(tc, Y2Z1, z, other.y, mnt4_modulus_device, MNT4_INV, 12);
  
  if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1) {
    // perform dbl case
    return dbl(tc);
  }

  // Add case:
  // if we have arrived here we are in the add case

  // const mnt4753_Fq Z1Z2 = (this->Z_) * (other.Z_);        // Z1Z2 = Z1*Z2
  mfq_t Z1Z2;
  mont_mul_64_lane(tc, Z1Z2, z, other.z, mnt4_modulus_device, MNT4_INV, 12);
  // const mnt4753_Fq u    = Y2Z1 - Y1Z2; // u    = Y2*Z1-Y1Z2
  mfq_t u;
  fq_sub_mod(tc, Y2Z1, Y1Z2, mnt4_modulus_device[tc.lane]);

  // const mnt4753_Fq uu   = u.squared();                  // uu   = u^2
  mfq_t uu = myfq_square(tc, u);

  // const mnt4753_Fq v    = X2Z1 - X1Z2; // v    = X2*Z1-X1Z2
  mfq_t v;
  fq_sub_mod(tc, X2Z1, X1Z2, mnt4_modulus_device[tc.lane]);

  // const mnt4753_Fq vv   = v.squared();                  // vv   = v^2
  mfq_t vv = myfq_square(tc, v);

  // const mnt4753_Fq vvv  = v * vv;                       // vvv  = v*vv
  mfq_t vvv;
  mont_mul_64_lane(tc, vvv, v, vv, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq R    = vv * X1Z2;                    // R    = vv*X1Z2
  mfq_t R;
  mont_mul_64_lane(tc, R, vv, X1Z2, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq A    = uu * Z1Z2 - (vvv + R + R);    // A    = uu*Z1Z2 - vvv - 2*R
  mfq_t A;
  mont_mul_64_lane(tc, A, uu, Z1Z2, mnt4_modulus_device, MNT4_INV, 12);
  fq_sub_mod(tc, A, vvv, mnt4_modulus_device[tc.lane]);
  fq_sub_mod(tc, A, R, mnt4_modulus_device[tc.lane]);
  fq_sub_mod(tc, A, R, mnt4_modulus_device[tc.lane]);

  // const mnt4753_Fq X3   = v * A;                        // X3   = v*A
  mfq_t X3;
  mont_mul_64_lane(tc, X3, v, A, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq Y3   = u * (R-A) - vvv * Y1Z2;       // Y3   = u*(R-A) - vvv*Y1Z2
  mfq_t Y3 = R;
  fq_sub_mod(tc, Y3, A, mnt4_modulus_device[tc.lane]);
  mont_mul_64_lane(tc, Y3, u, Y3, mnt4_modulus_device, MNT4_INV, 12);
  mfq_t second_term; // vvv * Y1Z2
  mont_mul_64_lane(tc, second_term, vvv, Y1Z2, mnt4_modulus_device, MNT4_INV, 12);
  fq_sub_mod(tc, Y3, second_term, mnt4_modulus_device[tc.lane]);
  
  // const mnt4753_Fq Z3   = vvv * Z1Z2;                   // Z3   = vvv*Z1Z2
  mfq_t Z3;
  mont_mul_64_lane(tc, Z3, vvv, Z1Z2, mnt4_modulus_device, MNT4_INV, 12);

  x = X3; 
  y = Y3;
  z = Z3;
}

__device__
void m4g1::dbl(thread_context_t& tc) {

  // TODO build
  if (is_zero(tc)) return;

  // const mnt4753_Fq XX   = (this->X_).squared();                   // XX  = X1^2
  mfq_t XX = myfq_square(tc, x);

  // const mnt4753_Fq ZZ   = (this->Z_).squared();                   // ZZ  = Z1^2
  mfq_t ZZ = myfq_square(tc, z);

  // const mnt4753_Fq w    = mnt4753_G1::coeff_a * ZZ + (XX + XX + XX); // w   = a*ZZ + 3*XX
  mfq_t aZZ;  
  mont_mul_64_lane(tc, aZZ, g1_mnt4_coeff_a[tc.lane], ZZ, mnt4_modulus_device, MNT4_INV, 12);
  mfq_t XX3 = XX;
  fq_mul_const_mod(tc, XX3, mnt4_modulus_device[tc.lane], 3);
  mfq_t w = aZZ;
  fq_add_mod(tc, w, XX3, mnt4_modulus_device[tc.lane]);

  // const mnt4753_Fq Y1Z1 = (this->Y_) * (this->Z_);
  mfq_t Y1Z1;
  mont_mul_64_lane(tc, Y1Z1, y, z, mnt4_modulus_device, MNT4_INV, 12);

  mfq_t s = Y1Z1;
  // const mnt4753_Fq s    = Y1Z1 + Y1Z1;                            // s   = 2*Y1*Z1
  fq_add_mod(tc, s, Y1Z1, mnt4_modulus_device[tc.lane]);

  // const mnt4753_Fq ss   = s.squared();                            // ss  = s^2
  mfq_t ss = myfq_square(tc, Y1Z1);

  // const mnt4753_Fq sss  = s * ss;                                 // sss = s*ss
  mfq_t sss;
  mont_mul_64_lane(tc, sss, ss, Y1Z1, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq R    = (this->Y_) * s;                         // R   = Y1*s
  mfq_t R;
  mont_mul_64_lane(tc, R, y, s, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq RR   = R.squared();                            // RR  = R^2
  mfq_t RR = myfq_square(tc, R);

  // const mnt4753_Fq B    = ((this->X_)+R).squared()-XX-RR;         // B   = (X1+R)^2 - XX - RR
  mfq_t B = x;
  fq_add_mod(tc, B, R, mnt4_modulus_device[tc.lane]);
  mfq_t BB = myfq_square(tc, B);
  fq_sub_mod(tc, BB, XX, mnt4_modulus_device[tc.lane]);
  fq_sub_mod(tc, BB, RR, mnt4_modulus_device[tc.lane]);

  // const mnt4753_Fq h    = w.squared() - (B+B);                    // h   = w^2 - 2*B
  mfq_t h = myfq_square(tc, w);
  fq_sub_mod(tc, h, B, mnt4_modulus_device[tc.lane]);
  fq_sub_mod(tc, h, B, mnt4_modulus_device[tc.lane]);

  // const mnt4753_Fq X3   = h * s;                                  // X3  = h*s
  mfq_t X3;
  mont_mul_64_lane(tc, X3, h, s, mnt4_modulus_device, MNT4_INV, 12);

  // const mnt4753_Fq Y3   = w * (B-h)-(RR+RR);                      // Y3  = w*(B-h) - 2*RR
  mfq_t RR2 = RR;
  fq_add_mod(tc, RR2, RR2, mnt4_modulus_device[tc.lane]);
  mfq_t B_h = B;
  fq_sub_mod(tc, B, h, mnt4_modulus_device[tc.lane]);
  mfq_t Y3 = w;
  mont_mul_64_lane(tc, Y3, w, B_h, mnt4_modulus_device, MNT4_INV, 12);
  fq_sub_mod(tc, Y3, RR2, mnt4_modulus_device[tc.lane]);

  // const mnt4753_Fq Z3   = sss;                                    // Z3  = sss
  mfq_t Z3 = sss;

   x = X3;
   y = Y3;
   z = Z3;
  
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

__device__ uint32_t fast_by2_ceil(uint32_t input) {
  return (input >> 1) + (input & 0x01);
}

__global__
void fq_g1mfq_add_kernel(g1mfq_ti* instances, uint32_t instance_count) {
  // Create an array of m4g1 objects.
  // Implement functions.
  thread_context_t tc_out;
  compute_context(tc_out, instance_count);
  if (tc.instance_number >= instance_count) return;
  //m4g1* my_instance = m4g1::get(&instances[tc.instance_number], tc);

  // num_output_instances captures the number of valid instances at the end of the loop.
  uint32_t num_output_instances = instance_count; 
  for (uint32_t s = 1; s < instance_count; s *= 2) {
     num_output_instances = fast_by2_ceil(num_output_instances);
     thread_context_t tc;
     compute_context(tc, num_output_instances);
     if (tc.instance_number >= num_output_instances) return;
     uint32_t index = tc.instance_number * 2;
     m4g1 my_instance(instances[index * s].x[tc.lane],
          instances[index* s].y[tc.lane],
          instances[(index + 1)*s].z[tc.lane]);
     m4g1 other(instances[(index + 1) * s].x[tc.lane],
          instances[(index + 1) * s].y[tc.lane],
          instances[(index + 1)*s].z[tc.lane]);
     my_instance.plusEquals(other);
  }
  if (tc_out.instance_number != 0) return;
  m4g1 my_instance(instances[0].x[tc.lane],
          instances[0].y[tc.lane],
          instances[0].z[tc.lane]);
  my_instance.to_affine();
}

__global__
void fq_g1mfq2_add_kernel(g1mfq2_ti* instances, uint32_t instance_count) {
  thread_context_t tc;
  compute_context(tc, instance_count);

  if (tc.instance_number >= instance_count) return;
  
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
