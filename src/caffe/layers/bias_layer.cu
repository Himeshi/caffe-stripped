#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"

namespace caffe {

template <typename Dtype>
__global__ void BiasForward(const int n, const fp16* in,
    const fp16* bias, const int bias_dim, const int inner_dim,
    fp16* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int bias_index = (index / inner_dim) % bias_dim;
    out[index] = fp32tofp16_gpu(fp16tofp32_gpu(in[index]) + fp16tofp32_gpu(bias[bias_index]));
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
      const vector<Blob<Dtype>*>& top_dtype) {
  const int count = top[0]->count();
  const fp16* bottom_data = bottom[0]->gpu_data();
  const fp16* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  fp16* top_data = top[0]->mutable_gpu_data();
  BiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bias_data, bias_dim_, inner_dim_, top_data);
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	  const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const fp16* top_diff = top[0]->gpu_diff();
    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const fp16* top_diff = top[0]->gpu_diff();
    fp16* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_gpu_diff();
    bool accum = bias_param;
    for (int n = 0; n < outer_dim_; ++n) {
      caffe_gpu_gemv_half(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
          top_diff, bias_multiplier_.gpu_data(), Dtype(accum), bias_diff);
      top_diff += dim_;
      accum = true;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BiasLayer);

}  // namespace caffe
