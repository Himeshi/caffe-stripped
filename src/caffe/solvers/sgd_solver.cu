#include "caffe/util/math_functions.hpp"

#include "caffe/fp16.cuh"

namespace caffe {

template <typename Dtype>
__global__ void SGDUpdate(int N, unsigned int* g, unsigned int* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = fp32tofp16_gpu(momentum*fp16tofp32_gpu(h[i]) + local_rate*fp16tofp32_gpu(g[i]));
  }
}
template <typename Dtype>
void sgd_update_gpu(int N, unsigned int* g, unsigned int* h, Dtype momentum,
    Dtype local_rate) {
  SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void sgd_update_gpu<float>(int, unsigned int*, unsigned int*, float, float);
template void sgd_update_gpu<double>(int, unsigned int*, unsigned int*, double, double);

}  // namespace caffe
