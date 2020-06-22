#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"

namespace caffe {

template <typename Dtype>
__global__ void NesterovUpdate(int N, FP16_TYPE* g, FP16_TYPE* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    FP16_TYPE hi = h[i];
    FP16_TYPE hi_new = h[i] = fp32tofp16_gpu(momentum * fp16tofp32_gpu(hi) + local_rate * fp16tofp32_gpu(g[i]));
    g[i] = fp32tofp16_gpu((1+momentum) * fp16tofp32_gpu(hi_new) - momentum * fp16tofp32_gpu(hi));
  }
}
template <typename Dtype>
void nesterov_update_gpu(int N, FP16_TYPE* g, FP16_TYPE* h, Dtype momentum,
    Dtype local_rate) {
  NesterovUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void nesterov_update_gpu<float>(int, FP16_TYPE*, FP16_TYPE*, float, float);
template void nesterov_update_gpu<double>(int, FP16_TYPE*, FP16_TYPE*, double,
    double);

}  // namespace caffe
