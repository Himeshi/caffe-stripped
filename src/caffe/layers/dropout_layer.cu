#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DropoutForward(const int n, const __half* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    __half* out) {
  CUDA_KERNEL_LOOP(index, n) {
    float temp_in = fp16tofp32_gpu(in[index]);
    float temp_out = temp_in * (mask[index] > threshold) * scale;
    out[index] = fp32tofp16_gpu(temp_out);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<__half>*>& bottom,
    const vector<Blob<__half>*>& top) {
  const __half* bottom_data = bottom[0]->gpu_data();
  __half* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const __half* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    __half* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    float temp_in = fp16tofp32_gpu(in_diff[index]);
    float temp_out = temp_in * scale * (mask[index] > threshold);
    out_diff[index] = fp32tofp16_gpu(temp_out);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<__half>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<__half>*>& bottom) {
  if (propagate_down[0]) {
    const __half* top_diff = top[0]->gpu_diff();
    __half* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
