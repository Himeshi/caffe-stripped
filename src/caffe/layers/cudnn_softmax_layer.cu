#ifdef USE_CUDNN
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
    const vector<Blob<Dtype>*>& top_dtype) {
  const fp16* bottom_data = bottom[0]->gpu_data();
  Blob<Dtype>* temp_bottom = (this->temp_bottom_);
  temp_bottom->Reshape(bottom[0]->shape());
  Dtype* temp_bottom_converted = temp_bottom->mutable_gpu_data();
  int bottom_count = bottom[0]->count();
  convert_to_float<<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_count, bottom_data, temp_bottom_converted);
  const Dtype* temp_bottom_data = temp_bottom->gpu_data();

  Blob<Dtype>* top_temp = (this->temp_top_);
  top_temp->Reshape(top[0]->shape());
  Dtype* top_data_temp = top_temp->mutable_gpu_data();

  CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, temp_bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data_temp));

  fp16* top_data = top[0]->mutable_gpu_data();
  int top_data_count = top[0]->count();
  convert_to_fp16<<<CAFFE_GET_BLOCKS(top_data_count), CAFFE_CUDA_NUM_THREADS>>>(top_data_count, top_data_temp, top_data);
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom) {
  if (propagate_down[0]) {

    const fp16* bottom_data = bottom[0]->gpu_data();
    Blob<Dtype>* temp_bottom = (this->temp_bottom_);
    temp_bottom->Reshape(bottom[0]->shape());
    Dtype* temp_bottom_converted = temp_bottom->mutable_gpu_data();
    int bottom_count = bottom[0]->count();
    convert_to_float<<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_count, bottom_data, temp_bottom_converted);
    const Dtype* temp_bottom_data = temp_bottom->gpu_data();

    Dtype* bottom_diff_temp = temp_bottom->mutable_gpu_diff();

    const fp16* top_diff = top[0]->gpu_diff();
    Blob<Dtype>* temp_top = (this->temp_top_);
    temp_top->Reshape(top[0]->shape());
    Dtype* temp_top_converted = temp_top->mutable_gpu_diff();
    int top_count = top[0]->count();
    convert_to_float<<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(top_count, top_diff, temp_top_converted);
    const Dtype* temp_top_diff = temp_top->gpu_diff();

    const fp16* top_data = top[0]->gpu_data();
    Dtype* temp_top_data_converted = temp_top->mutable_gpu_data();
    convert_to_float<<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(top_count, top_data, temp_top_data_converted);
    const Dtype* temp_top_data = temp_top->gpu_data();

    CUDNN_CHECK(cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_CHANNEL,
          cudnn::dataType<Dtype>::one,
          top_desc_, temp_top_data, top_desc_, temp_top_diff,
          cudnn::dataType<Dtype>::zero,
          bottom_desc_, bottom_diff_temp));

    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    convert_to_fp16<<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_count, bottom_diff_temp, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNSoftmaxLayer);

}  // namespace caffe
#endif
