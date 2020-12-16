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
  caffe_expand_blob_w(weight_count, weight_temp, weight, this->blobs_[0]->data_bias);
  const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    int bottom_data_count;
    Dtype* bottom_data_dtype;
    if(this->layer_param_.name() == "conv1") {
      bottom_data_count = bottom_dtype[i]->count();
      bottom_data_dtype = bottom_dtype[i]->mutable_gpu_data();
    } else  {
      bottom_data_count = bottom[i]->count();
      const fp16* bottom_data = bottom[i]->gpu_data();
      this->temp_bottom_->Reshape(bottom[i]->shape());
      bottom_data_dtype = this->temp_bottom_->mutable_gpu_data();
      caffe_expand_blob(bottom_data_count, bottom_data_dtype, bottom_data, bottom[i]->data_bias);
    }
    fp16* top_data = top[i]->mutable_gpu_data();
    this->temp_top_->Reshape(top[i]->shape());
    Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();
    caffe_expand_blob(top[i]->count(), temp_top_data, top_data, top[i]->data_bias);

    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data_dtype + n * this->bottom_dim_, weight_temp_data,
          temp_top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const fp16* bias = this->blobs_[1]->gpu_data();
        Dtype* bias_temp = this->blobs_dtype_[1]->mutable_gpu_data();
        int bias_count = this->blobs_[1]->count();
        caffe_expand_blob_w(bias_count, bias_temp, bias, this->blobs_[1]->data_bias);
        const Dtype* bias_temp_data = this->blobs_dtype_[1]->gpu_data();
        this->forward_gpu_bias(temp_top_data + n * this->top_dim_, bias_temp_data);
      }
    }
    caffe_compress_blob(top[i]->count(), temp_top_data, top_data, &(top[i]->data_bias));
#ifdef SAMPLE_FLOATS
    if(this->phase_ == TRAIN && this->sample_iter_) {
      sample_blob(top[i]->gpu_data(), top[i]->count(), this->activation_exp, this->activation_frac, this->activation, this->activation_vector, SAMPLING_FREQ);
    }
#endif
  }
#ifdef SAMPLE_FLOATS
  if(this->phase_ == TRAIN && this->sample_iter_) {
    sample_blob(weight, this->blobs_[0]->count(), this->weight_exp, this->weight_frac, this->weight, this->weight_vector, WEIGHT_SAMPLING_FREQ);
    sample_blob(this->blobs_[1]->gpu_data(), this->blobs_[1]->count(), this->bias_exp, this->bias_frac, this->bias, this->bias_vector, BIAS_SAMPLING_FREQ);
  }
#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	  const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  const fp16* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  int weight_count = this->blobs_[0]->count();
  caffe_expand_blob_w(weight_count, weight_temp, weight, this->blobs_[0]->data_bias);
  const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

  fp16* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* temp_weight_diff = this->blobs_dtype_[0]->mutable_gpu_diff();
  caffe_expand_blob(weight_count, temp_weight_diff, weight_diff, this->blobs_[0]->diff_bias);

  for (int i = 0; i < top.size(); ++i) {
	const fp16* top_diff = top[i]->gpu_diff();
    this->temp_top_->Reshape(top[i]->shape());
    Dtype* temp_top_diff = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob(top[i]->count(), temp_top_diff, top_diff, top[i]->diff_bias);

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      fp16* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      Dtype* temp_bias_diff = this->blobs_dtype_[1]->mutable_gpu_diff();
      caffe_expand_blob(this->blobs_dtype_[1]->count(), temp_bias_diff, bias_diff, this->blobs_[1]->diff_bias);
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(temp_bias_diff, temp_top_diff + n * this->top_dim_);
      }
      caffe_compress_blob(this->blobs_[1]->count(), temp_bias_diff, bias_diff, &((this->blobs_[1])->diff_bias));
#ifdef SAMPLE_FLOATS
    if(this->phase_ == TRAIN && this->sample_iter_) {
      sample_blob(this->blobs_[1]->gpu_diff(), this->blobs_[1]->count(), this->bias_gradient_exp, this->bias_gradient_frac, this->bias_gradient, this->bias_gradient_vector, SAMPLING_FREQ);
    }
#endif
    }

    if (this->param_propagate_down_[0] || propagate_down[i]) {
      int bottom_count = bottom[i]->count();
      Dtype* temp_bottom_data;
      this->temp_bottom_->Reshape(bottom[i]->shape());
      if(this->layer_param_.name() == "conv1") {
        temp_bottom_data = bottom_dtype[i]->mutable_gpu_data();
      } else {
        const fp16* bottom_data = bottom[i]->gpu_data();
        temp_bottom_data = this->temp_bottom_->mutable_gpu_data();
        caffe_expand_blob(bottom_count, temp_bottom_data, bottom_data, bottom[i]->data_bias);
      }

      fp16* bottom_diff = bottom[i]->mutable_gpu_diff();
      Dtype* temp_bottom_diff = this->temp_bottom_->mutable_gpu_diff();
      caffe_expand_blob(bottom_count, temp_bottom_diff, bottom_diff, bottom[i]->diff_bias);
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(temp_bottom_data + n * this->bottom_dim_,
            temp_top_diff + n * this->top_dim_, temp_weight_diff);
          caffe_compress_blob(this->blobs_[0]->count(), temp_weight_diff, weight_diff, &((this->blobs_[0])->diff_bias));
#ifdef SAMPLE_FLOATS
  if(this->phase_ == TRAIN && this->sample_iter_) {
    sample_blob(this->blobs_[0]->gpu_diff(), this->blobs_[0]->count(), this->weight_gradient_exp, this->weight_gradient_frac, this->weight_gradient, this->weight_gradient_vector, SAMPLING_FREQ);
  }
#endif
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(temp_top_diff + n * this->top_dim_, weight_temp_data,
            temp_bottom_diff + n * this->bottom_dim_);
          caffe_compress_blob(bottom_count, temp_bottom_diff, bottom_diff, &(bottom[i]->diff_bias));
#ifdef SAMPLE_FLOATS
      if(this->phase_ == TRAIN && this->sample_iter_) {
        sample_blob(bottom[i]->gpu_diff(), bottom[i]->count(), this->activation_gradient_exp, this->activation_gradient_frac, this->activation_gradient, this->activation_gradient_vector, SAMPLING_FREQ);
      }
#endif
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
