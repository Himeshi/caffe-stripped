#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top) {

  const fp16* bottom_data = bottom[0]->gpu_data();
  Blob<Dtype>* temp_bottom = (this->temp_bottom_);
  temp_bottom->Reshape(bottom[0]->shape());
  Dtype* temp_bottom_converted = temp_bottom->mutable_gpu_data();
  int bottom_count = bottom[0]->count();
  convert_to_float<<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_count, bottom_data, temp_bottom_converted);
  const Dtype* temp_bottom_data = temp_bottom->gpu_data();

  const fp16* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  int weight_count = this->blobs_[0]->count();
  convert_to_float<<<CAFFE_GET_BLOCKS(weight_count), CAFFE_CUDA_NUM_THREADS>>>(weight_count, weight, weight_temp);
  const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

  const fp16* bias_data = this->blobs_[1]->gpu_data();
  Dtype* bias_temp = this->blobs_dtype_[1]->mutable_gpu_data();
  int bias_count = this->blobs_[1]->count();
  convert_to_float<<<CAFFE_GET_BLOCKS(bias_count), CAFFE_CUDA_NUM_THREADS>>>(bias_count, bias_data, bias_temp);
  const Dtype* bias_temp_data = this->blobs_dtype_[1]->gpu_data();

  Blob<Dtype>* top_temp = (this->temp_top_);
  top_temp->Reshape(top[0]->shape());
  Dtype* top_data_temp = top_temp->mutable_gpu_data();

  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight_temp_data, temp_bottom_data, (Dtype)0., top_data_temp);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            bias_temp_data, top_data_temp);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          temp_bottom_data, weight_temp_data, (Dtype)0., top_data_temp);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            bias_temp_data, (Dtype)1., top_data_temp);
  }
  fp16* top_data = top[0]->mutable_gpu_data();
  int top_data_count = top[0]->count();
  convert_to_fp16<<<CAFFE_GET_BLOCKS(top_data_count), CAFFE_CUDA_NUM_THREADS>>>(top_data_count, top_data_temp, top_data);
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
    caffe_gpu_gemv_half<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
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
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
