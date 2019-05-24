#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::Forward_gpu(const vector<Blob<__half>*>& bottom,
    const vector<Blob<__half>*>& top) {
  const __half* bottom_data = bottom[0]->gpu_data();
  __half* top_data = top[0]->mutable_gpu_data();

  CUDNN_CHECK(cudnnDivisiveNormalizationForward(
        handle_, norm_desc_, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        NULL,  // srcMeansData
        this->tempData1, this->tempData2,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data) );
}

template <typename Dtype>
void CuDNNLCNLayer<Dtype>::Backward_gpu(const vector<Blob<__half>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<__half>*>& bottom) {
  const __half* top_diff = top[0]->gpu_diff();
  const __half* top_data = top[0]->gpu_data();
  const __half* bottom_data = bottom[0]->gpu_data();
  __half* bottom_diff = bottom[0]->mutable_gpu_diff();

  CUDNN_CHECK(cudnnDivisiveNormalizationBackward(
        handle_, norm_desc_, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        NULL, top_diff,  // NULL - srcMeansData
        this->tempData1, this->tempData2,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff,
        NULL) );
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNLCNLayer);

}  // namespace caffe
#endif
