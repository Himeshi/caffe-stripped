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

  fp16* top_data = top[0]->mutable_gpu_data();

  const fp16* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  int weight_count = this->blobs_[0]->count();
  caffe_expand_blob(weight_count, weight_temp, weight, this->blobs_[0]->data_bias);
  const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

  if (M_ == 1) {
	  caffe_gpu_gemv_with_float_weights(CblasNoTrans, N_, K_, (1.),
    		weight_temp_data, bottom_data, (0.), top_data);
    if (bias_term_)
      caffe_gpu_axpy(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
	  caffe_gpu_gemm_half_with_floatB(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, Dtype(1.),
						  bottom_data, weight_temp_data, Dtype(0.), top_data);
    if (bias_term_)
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, fp32tofp16(1.),
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), fp32tofp16(1.), top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
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
    const fp16* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
    int weight_count = this->blobs_[0]->count();
    caffe_expand_blob(weight_count, weight_temp, weight, this->blobs_[0]->data_bias);
    const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

    if (transpose_) {
      caffe_gpu_gemm_half_with_floatB(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight_temp_data,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm_half_with_floatB(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, weight_temp_data,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
