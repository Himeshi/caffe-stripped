#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] = data[index] - channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract_backward(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, fp16* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] = fp32tofp16_gpu(fp16tofp32_gpu(data[index]) - channel_max[n * spatial_dim + s]);
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] = data[index] / channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const fp16* data_1, const fp16* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (fp16tofp32_gpu(data_1[(n * channels + c) * spatial_dim + s])
          * fp16tofp32_gpu(data_2[(n * channels + c) * spatial_dim + s]));
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
    const vector<Blob<Dtype>*>& top_dtype) {
  const fp16* bottom_data = bottom[0]->gpu_data();
  Blob<Dtype>* temp_bottom = (this->temp_bottom_);
  temp_bottom->Reshape(bottom[0]->shape());
  Dtype* temp_bottom_converted = temp_bottom->mutable_gpu_data();
  int count = bottom[0]->count();
  caffe_expand_blob(count, temp_bottom_converted, bottom_data, bottom[0]->data_bias);
  const Dtype* temp_bottom_data = temp_bottom->gpu_data();

  fp16* top_data = top[0]->mutable_gpu_data();
  (this->temp_top_)->Reshape(top[0]->shape());
  Dtype* temp_top_data = (this->temp_top_)->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, temp_bottom_data, temp_top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, temp_top_data,
      scale_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, temp_top_data);
  caffe_compress_blob(count, temp_top_data, top_data, &(top[0]->data_bias));
  caffe_expand_blob(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);

  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, temp_top_data, temp_top_data);
  caffe_compress_blob(count, temp_top_data, top_data, &(top[0]->data_bias));
  caffe_expand_blob(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);

  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, temp_top_data,
      scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, temp_top_data);

  caffe_compress_blob(count, temp_top_data, top_data, &(top[0]->data_bias));
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  const fp16* top_diff = top[0]->gpu_diff();
  const fp16* top_data = top[0]->gpu_data();
  fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int count = top[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, top_diff, bottom_diff);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_,
      top_diff, top_data, scale_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract_backward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, bottom_diff);
  // elementwise multiplication
  caffe_gpu_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);


}  // namespace caffe
