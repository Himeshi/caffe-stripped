#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
    const vector<Blob<Dtype>*>& top_dtype) {
  const fp16* bottom_data = bottom[0]->gpu_data();
  int bottom_data_count = bottom[0]->count();
  this->temp_bottom_->Reshape(bottom[0]->shape());
  Dtype* bottom_data_dtype = this->temp_bottom_->mutable_gpu_data();
  caffe_expand_blob_activations(bottom_data_count, bottom_data_dtype, bottom_data, bottom[0]->data_bias);

  fp16* top_data = top[0]->mutable_gpu_data();
  this->temp_top_->Reshape(top[0]->shape());
  Dtype* temp_top_data = this->temp_top_->mutable_gpu_data();
  caffe_expand_blob_activations(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);

  const fp16* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  int weight_count = this->blobs_[0]->count();
  caffe_expand_blob_w(weight_count, weight_temp, weight, this->blobs_[0]->data_bias);
  const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

  if (M_ == 1) {
	  caffe_gpu_gemv(CblasNoTrans, N_, K_, Dtype(1.),
        weight_temp_data, bottom_data_dtype, Dtype(0.), temp_top_data);
	caffe_compress_blob_activations(top[0]->count(), temp_top_data, top_data, &(top[0]->data_bias));
	caffe_expand_blob_activations(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);
    if (bias_term_) {
      const fp16* bias = this->blobs_[1]->gpu_data();
      Dtype* bias_temp = this->blobs_dtype_[1]->mutable_gpu_data();
      int bias_count = this->blobs_[1]->count();
      caffe_expand_blob_w(bias_count, bias_temp, bias, this->blobs_[1]->data_bias);
      const Dtype* bias_temp_data = this->blobs_dtype_[1]->gpu_data();
      caffe_gpu_axpy(N_, bias_multiplier_.cpu_data()[0],
                       bias_temp_data, temp_top_data);
    }
  } else {
	  caffe_gpu_gemm(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, Dtype(1.),
						  bottom_data_dtype, weight_temp_data, Dtype(0.), temp_top_data);
	caffe_compress_blob_activations(top[0]->count(), temp_top_data, top_data, &(top[0]->data_bias));
	caffe_expand_blob_activations(top[0]->count(), temp_top_data, top_data, top[0]->data_bias);
    if (bias_term_) {
      const fp16* bias = this->blobs_[1]->gpu_data();
      Dtype* bias_temp = this->blobs_dtype_[1]->mutable_gpu_data();
      int bias_count = this->blobs_[1]->count();
      caffe_expand_blob_w(bias_count, bias_temp, bias, this->blobs_[1]->data_bias);
      const Dtype* bias_temp_data = this->blobs_dtype_[1]->gpu_data();
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1.),
                            bias_multiplier_.gpu_data(),
							bias_temp_data, Dtype(1.), temp_top_data);
    }
  }
  caffe_compress_blob_activations(top[0]->count(), temp_top_data, top_data, &(top[0]->data_bias));
#ifdef SAMPLE_FLOATS
    if(this->phase_ == TRAIN && this->sample_iter_) {
      sample_blob(top[0]->gpu_data(), top[0]->count(), this->activation_exp, this->activation_frac, this->activation, this->activation_vector, SAMPLING_FREQ);
      sample_blob(weight, this->blobs_[0]->count(), this->weight_exp, this->weight_frac, this->weight, this->weight_vector, WEIGHT_SAMPLING_FREQ);
      sample_blob(this->blobs_[1]->gpu_data(), this->blobs_[1]->count(), this->bias_exp, this->bias_frac, this->bias, this->bias_vector, BIAS_SAMPLING_FREQ);
    }
#endif
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  if (this->param_propagate_down_[0]) {
    const fp16* top_diff = top[0]->gpu_diff();
    int top_count = top[0]->count();
    this->temp_top_->Reshape(top[0]->shape());
    Dtype* top_diff_dtype = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_ag(top_count, top_diff_dtype, top_diff, top[0]->diff_bias);

    const fp16* bottom_data = bottom[0]->gpu_data();
    int bottom_data_count = bottom[0]->count();
    this->temp_bottom_->Reshape(bottom[0]->shape());
    Dtype* bottom_data_dtype = this->temp_bottom_->mutable_gpu_data();
    caffe_expand_blob_activations(bottom_data_count, bottom_data_dtype, bottom_data, bottom[0]->data_bias);

    fp16* weight = this->blobs_[0]->mutable_gpu_diff();
    Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_diff();
    caffe_expand_blob(this->blobs_dtype_[0]->count(), weight_temp, weight, this->blobs_[0]->diff_bias);

    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data_dtype, top_diff_dtype,
          (Dtype)1., weight_temp);
    } else {
      caffe_gpu_gemm(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff_dtype, bottom_data_dtype,
          (Dtype)1., weight_temp);
    }
    caffe_compress_blob((this->blobs_[0])->count(), weight_temp, weight, &((this->blobs_[0])->diff_bias));
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const fp16* top_diff = top[0]->gpu_diff();
    int top_count = top[0]->count();
    this->temp_top_->Reshape(top[0]->shape());
    Dtype* top_diff_dtype = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_ag(top_count, top_diff_dtype, top_diff, top[0]->diff_bias);

    fp16* bias = this->blobs_[1]->mutable_gpu_diff();
    Dtype* bias_temp = this->blobs_dtype_[1]->mutable_gpu_diff();
    caffe_expand_blob(this->blobs_dtype_[1]->count(), bias_temp, bias, this->blobs_[1]->diff_bias);

    // Gradient with respect to bias
    caffe_gpu_gemv(CblasTrans, M_, N_, Dtype(1.), top_diff_dtype,
        bias_multiplier_.gpu_data(), Dtype(1.),
		bias_temp);
    caffe_compress_blob((this->blobs_[1])->count(), bias_temp, bias, &((this->blobs_[1])->diff_bias));
  }
  if (propagate_down[0]) {
    const fp16* top_diff = top[0]->gpu_diff();
    int top_count = top[0]->count();
    this->temp_top_->Reshape(top[0]->shape());
    Dtype* top_diff_dtype = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_ag(top_count, top_diff_dtype, top_diff, top[0]->diff_bias);

    // Gradient with respect to bottom data
    const fp16* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
    int weight_count = this->blobs_[0]->count();
    caffe_expand_blob_w(weight_count, weight_temp, weight, this->blobs_[0]->data_bias);
    const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    this->temp_bottom_->Reshape(bottom[0]->shape());
    Dtype* bottom_diff_temp = this->temp_bottom_->mutable_gpu_diff();
    caffe_expand_blob_ag(bottom[0]->count(), bottom_diff_temp, bottom_diff, bottom[0]->diff_bias);

    if (transpose_) {
      caffe_gpu_gemm(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff_dtype, weight_temp_data,
          (Dtype)0., bottom_diff_temp);
    } else {
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff_dtype, weight_temp_data,
         (Dtype)0., bottom_diff_temp);
    }
    caffe_compress_blob_ag(bottom[0]->count(), bottom_diff_temp, bottom_diff, &(bottom[0]->diff_bias));
  }
#ifdef SAMPLE_FLOATS
  if(this->phase_ == TRAIN && this->sample_iter_) {
    sample_blob(this->blobs_[0]->gpu_diff(), this->blobs_[0]->count(), this->weight_gradient_exp, this->weight_gradient_frac, this->weight_gradient, this->weight_gradient_vector, SAMPLING_FREQ);
    sample_blob(this->blobs_[1]->gpu_diff(), this->blobs_[1]->count(), this->bias_gradient_exp, this->bias_gradient_frac, this->bias_gradient, this->bias_gradient_vector, SAMPLING_FREQ);
    sample_blob(bottom[0]->gpu_diff(), bottom[0]->count(), this->activation_gradient_exp, this->activation_gradient_frac, this->activation_gradient, this->activation_gradient_vector, SAMPLING_FREQ);
  }
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
