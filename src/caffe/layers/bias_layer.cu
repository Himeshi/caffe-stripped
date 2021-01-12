#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"

namespace caffe {

template <typename Dtype>
__global__ void BiasForward(const int n, const Dtype* in,
    const Dtype* bias, const int bias_dim, const int inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int bias_index = (index / inner_dim) % bias_dim;
    out[index] = in[index] + bias[bias_index];
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
      const vector<Blob<Dtype>*>& top_dtype) {
  const int count = top[0]->count();
  const fp16* bottom_data = bottom[0]->gpu_data();
  Dtype* bias_temp;
  if(bottom.size() > 1) {
	  this->blobs_dtype_[0]->Reshape(bottom[1]->shape());
	  bias_temp = this->blobs_dtype_[0]->mutable_gpu_data();
	  int bias_count = bottom[1]->count();
	  caffe_expand_blob_activations(bias_count, bias_temp, bottom[1]->gpu_data(), bottom[1]->data_bias);
  } else {
	  this->blobs_dtype_[0]->Reshape(this->blobs_[0]->shape());
	  bias_temp = this->blobs_dtype_[0]->mutable_gpu_data();
	  int bias_count = this->blobs_[0]->count();
	  caffe_expand_blob_w(bias_count, bias_temp, this->blobs_[0]->gpu_data(), this->blobs_[0]->data_bias);
  }

  fp16* top_data = top[0]->mutable_gpu_data();

  this->temp_bottom_->Reshape(bottom[0]->shape());
  Dtype* bottom_data_dtype = this->temp_bottom_->mutable_gpu_data();
  caffe_expand_blob_activations(bottom[0]->count(), bottom_data_dtype, bottom_data, bottom[0]->data_bias);

  this->temp_top_->Reshape(top[0]->shape());
  Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();

  BiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_dtype, bias_temp, bias_dim_, inner_dim_, temp_top_data);

  caffe_compress_blob_activations(count, temp_top_data, top_data, &(top[0]->data_bias));
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	  const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const fp16* top_diff = top[0]->gpu_diff();
    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
    bottom[0]->diff_bias = top[0]->diff_bias;
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const fp16* top_diff = top[0]->gpu_diff();
    int top_count = top[0]->count();
    this->temp_top_->Reshape(top[0]->shape());
    Dtype* top_diff_dtype = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_ag(top_count, top_diff_dtype, top_diff, top[0]->diff_bias);

    fp16* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_gpu_diff();
    Dtype* bias_diff_temp;
    if(bias_param) {
  	  this->blobs_dtype_[0]->Reshape(this->blobs_[0]->shape());
  	  bias_diff_temp = this->blobs_dtype_[0]->mutable_gpu_diff();
  	  int bias_count = this->blobs_[0]->count();
  	  caffe_expand_blob(bias_count, bias_diff_temp, this->blobs_[0]->gpu_diff(), this->blobs_[0]->diff_bias);
    } else {
      this->blobs_dtype_[0]->Reshape(bottom[1]->shape());
      bias_diff_temp = this->blobs_dtype_[0]->mutable_gpu_diff();
      int bias_count = bottom[1]->count();
      caffe_expand_blob_ag(bias_count, bias_diff_temp, bottom[1]->gpu_diff(), bottom[1]->diff_bias);
    }
    bool accum = bias_param;
    for (int n = 0; n < outer_dim_; ++n) {
      caffe_gpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
        top_diff_dtype, bias_multiplier_.gpu_data(), Dtype(accum), bias_diff_temp);
      top_diff_dtype += dim_;
      accum = true;
    }
    if(bias_param) {
      caffe_compress_blob(this->blobs_[0]->count(), bias_diff_temp, bias_diff, &(this->blobs_[0]->diff_bias));
    } else {
      caffe_compress_blob_ag(bottom[1]->count(), bias_diff_temp, bias_diff, &(bottom[1]->diff_bias));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BiasLayer);

}  // namespace caffe
