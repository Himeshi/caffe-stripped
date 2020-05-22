#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& top_dtype) {
#ifdef CONVERT_SHARED
  const fp16* bottom_data = bottom[0]->gpu_data();
  Blob<Dtype>* temp_bottom = (this->temp_bottom_);
  temp_bottom->Reshape(bottom[0]->shape());
  Dtype* temp_bottom_converted = temp_bottom->mutable_gpu_data();
  int bottom_count = bottom[0]->count();
  convert_to_float<<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_count, bottom_data, temp_bottom_converted);
  const Dtype* temp_bottom_data = temp_bottom->gpu_data();

  fp16* top_data = top[0]->mutable_gpu_data();

  const fp16* weight = this->blobs_[0]->gpu_data();

  if (M_ == 1) {
    caffe_gpu_gemv(CblasNoTrans, N_, K_, fp32tofp16(1.),
                         weight, bottom_data, fp32tofp16(0.), top_data);
    if (bias_term_)
      caffe_gpu_axpy(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm_half_with_float(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, Dtype(1.),
                          temp_bottom_data, weight, Dtype(0.), top_data);
    if (bias_term_)
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, fp32tofp16(1.),
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), fp32tofp16(1.), top_data);
  }
#else
  const fp16* bottom_data = bottom[0]->gpu_data();
  fp16* top_data = top[0]->mutable_gpu_data();
  const fp16* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv(CblasNoTrans, N_, K_, fp32tofp16(1.),
                         weight, bottom_data, fp32tofp16(0.), top_data);
    if (bias_term_)
      caffe_gpu_axpy(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, fp32tofp16(1.),
                          bottom_data, weight, fp32tofp16(0.), top_data);
    if (bias_term_)
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, fp32tofp16(1.),
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), fp32tofp16(1.), top_data);
  }
#endif

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
    const vector<bool>& propagate_down,
    const vector<Blob<fp16>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const fp16* top_diff = top[0]->gpu_diff();
    const fp16* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm_half(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm_half(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const fp16* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv(CblasTrans, M_, N_, fp32tofp16(1.), top_diff,
        bias_multiplier_.gpu_data(), fp32tofp16(1.),
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const fp16* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm_half(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm_half(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
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
