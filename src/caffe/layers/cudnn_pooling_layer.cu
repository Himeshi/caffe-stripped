#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
    const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& top_dtype) {
  const fp16* bottom_data = bottom[0]->gpu_data();
  fp16* top_data = top[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnPoolingForward(handle_, pooling_desc_,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const fp16* top_diff = top[0]->gpu_diff();
  const fp16* top_data = top[0]->gpu_data();
  const fp16* bottom_data = bottom[0]->gpu_data();
  fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
  CUDNN_CHECK(cudnnPoolingBackward(handle_, pooling_desc_,
        cudnn::dataType<Dtype>::one,
        top_desc_, top_data, top_desc_, top_diff,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff));
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNPoolingLayer);

}  // namespace caffe
#endif
