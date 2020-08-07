#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
      const vector<Blob<Dtype>*>& top_dtype) {

  const fp16* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  int weight_count = this->blobs_[0]->count();
  convert_to_float<<<CAFFE_GET_BLOCKS(weight_count), CAFFE_CUDA_NUM_THREADS>>>(weight_count, weight, weight_temp);
  const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    if(this->layer_param_.name() == "conv1") {
      int bottom_data_count = bottom_dtype[i]->count();
      fp16* bottom_data_temp = bottom[i]->mutable_gpu_data();
      Dtype* bottom_data_dtype = bottom_dtype[i]->mutable_gpu_data();
      convert_to_fp16<<<CAFFE_GET_BLOCKS(bottom_data_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_data_count, bottom_data_dtype, bottom_data_temp);
    }
    const fp16* bottom_data = bottom[i]->gpu_data();
    fp16* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm_half_with_float_weights(bottom_data + n * this->bottom_dim_, weight_temp_data,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const fp16* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias_half(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	  const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
#ifdef CONVERT_SHARED
  const fp16* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  int weight_count = this->blobs_[0]->count();
  convert_to_float<<<CAFFE_GET_BLOCKS(weight_count), CAFFE_CUDA_NUM_THREADS>>>(weight_count, weight, weight_temp);
  const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

  fp16* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const fp16* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      fp16* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias_half(bias_diff, top_diff + n * this->top_dim_);
      }
#ifdef SAMPLE_FLOATS
    if(this->phase_ == TRAIN && this->sample_iter_) {
      sample_blob(this->blobs_[1]->gpu_diff(), this->blobs_[1]->count(), this->bias_gradient_exp, this->bias_gradient_frac, this->bias_gradient, this->bias_gradient_vector, SAMPLING_FREQ);
    }
#endif
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const fp16* bottom_data = bottom[i]->gpu_data();
      fp16* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm_half(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
#ifdef SAMPLE_FLOATS
  if(this->phase_ == TRAIN && this->sample_iter_) {
    sample_blob(this->blobs_[0]->gpu_diff(), this->blobs_[0]->count(), this->weight_gradient_exp, this->weight_gradient_frac, this->weight_gradient, this->weight_gradient_vector, SAMPLING_FREQ);
  }
#endif
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm_half_with_float(top_diff + n * this->top_dim_, weight_temp_data,
              bottom_diff + n * this->bottom_dim_);
#ifdef SAMPLE_FLOATS
      if(this->phase_ == TRAIN && this->sample_iter_) {
        sample_blob(bottom[i]->gpu_diff(), bottom[i]->count(), this->activation_gradient_exp, this->activation_gradient_frac, this->activation_gradient, this->activation_gradient_vector, SAMPLING_FREQ);
      }
#endif
        }
      }
    }
  }
#else
  const fp16* weight = this->blobs_[0]->gpu_data();
  fp16* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const fp16* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      fp16* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias_half(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const fp16* bottom_data = bottom[i]->gpu_data();
      fp16* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm_half(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm_half(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
