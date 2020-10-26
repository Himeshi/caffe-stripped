#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
    const vector<Blob<Dtype>*>& top_dtype) {
  const fp16* bottom_data = bottom[0]->gpu_data();
  const int count = bottom[0]->count();
  this->temp_bottom_->Reshape(bottom[0]->shape());
  Dtype* bottom_data_dtype = this->temp_bottom_->mutable_gpu_data();
  caffe_expand_blob(count, bottom_data_dtype, bottom_data, bottom[0]->data_bias);

  fp16* top_data = top[0]->mutable_gpu_data();
  this->temp_top_->Reshape(top[0]->shape());
  Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();

  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_dtype, temp_top_data, negative_slope);
  caffe_compress_blob(top[0]->count(), temp_top_data, top_data, &(top[0]->data_bias));
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
    const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const fp16* bottom_data = bottom[0]->gpu_data();
    this->temp_bottom_->Reshape(bottom[0]->shape());
    Dtype* bottom_data_dtype = this->temp_bottom_->mutable_gpu_data();
    caffe_expand_blob(count, bottom_data_dtype, bottom_data, bottom[0]->data_bias);

    const int top_count = top[0]->count();
    const fp16* top_diff = top[0]->gpu_diff();
    this->temp_top_->Reshape(top[0]->shape());
    Dtype* top_diff_dtype = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_bwd(top_count, top_diff_dtype, top_diff, top[0]->diff_bias);

    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff_dtype = this->temp_bottom_->mutable_gpu_diff();
    caffe_expand_blob_bwd(bottom[0]->count(), bottom_diff_dtype, bottom_diff, bottom[0]->diff_bias);

    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff_dtype, bottom_data_dtype, bottom_diff_dtype, negative_slope);
    caffe_compress_blob_bwd(count, bottom_diff_dtype, bottom_diff, &(bottom[0]->diff_bias));
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
