#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<__half>*>& bottom,
      const vector<Blob<__half>*>& top) {

  const __half* weight = this->blobs_[0]->gpu_data();
  int weight_count = this->blobs_[0]->count();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  convert_to_float<<<CAFFE_GET_BLOCKS(weight_count), CAFFE_CUDA_NUM_THREADS>>>(weight_count, weight, weight_temp);
  const Dtype* weight_converted = this->blobs_dtype_[0]->gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {

    const __half* bottom_data = bottom[i]->gpu_data();
    Blob<Dtype>* bottom_temp = &(this->temp_bottom_);
    bottom_temp->Reshape(bottom[i]->shape());
    Dtype* bottom_data_temp = bottom_temp->mutable_gpu_data();
    int bottom_data_count = top[i]->count();
    convert_to_float<<<CAFFE_GET_BLOCKS(bottom_data_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_data_count, bottom_data, bottom_data_temp);
    const Dtype* bottom_data_converted = bottom_temp->gpu_data();

    Blob<Dtype>* top_temp = &(this->temp_top_);
    top_temp->Reshape(top[i]->shape());
    Dtype* top_data_converted = top_temp->mutable_gpu_data();

    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data_converted + n * this->bottom_dim_, weight_converted,
          top_data_converted + n * this->top_dim_);

      if (this->bias_term_) {
        const __half* bias = this->blobs_[1]->gpu_data();
        int bias_count = this->blobs_[1]->count();
        Dtype* bias_temp = this->blobs_dtype_[1]->mutable_gpu_data();
        convert_to_float<<<CAFFE_GET_BLOCKS(bias_count), CAFFE_CUDA_NUM_THREADS>>>(bias_count, bias, bias_temp);

        this->forward_gpu_bias(top_data_converted + n * this->top_dim_, bias_temp);
      }

    }
    __half* top_data = top[i]->mutable_gpu_data();
    int top_data_count = bottom[i]->count();
    convert_to_fp16<<<CAFFE_GET_BLOCKS(top_data_count), CAFFE_CUDA_NUM_THREADS>>>(top_data_count, top_data_converted, top_data);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<__half>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<__half>*>& bottom) {

  const __half* weight = this->blobs_[0]->gpu_data();
  int weight_count = this->blobs_[0]->count();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  convert_to_float<<<CAFFE_GET_BLOCKS(weight_count), CAFFE_CUDA_NUM_THREADS>>>(weight_count, weight, weight_temp);
  const Dtype* weight_converted = this->blobs_dtype_[0]->gpu_data();

  Dtype* weight_diff_temp = this->blobs_dtype_[0]->mutable_gpu_diff();

  for (int i = 0; i < top.size(); ++i) {

    const __half* top_diff = top[i]->gpu_diff();
    int top_diff_count = top[i]->count();
    Blob<Dtype>* top_temp = &(this->temp_top_);
    top_temp->Reshape(top[i]->shape());
    Dtype* top_temp_diff = top_temp->mutable_gpu_diff();
    convert_to_float<<<CAFFE_GET_BLOCKS(top_diff_count), CAFFE_CUDA_NUM_THREADS>>>(top_diff_count, top_diff, top_temp_diff);
    const Dtype* top_diff_converted = top_temp->gpu_data();

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff_temp = this->blobs_dtype_[1]->mutable_gpu_diff();

      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff_temp, top_diff_converted + n * this->top_dim_);
      }

      __half* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      int bias_diff_count = this->blobs_[1]->count();
      convert_to_fp16<<<CAFFE_GET_BLOCKS(bias_diff_count), CAFFE_CUDA_NUM_THREADS>>>(bias_diff_count, bias_diff_temp, bias_diff);
    }

    if (this->param_propagate_down_[0] || propagate_down[i]) {

      Blob<Dtype>* bottom_temp = &(this->temp_bottom_);
      bottom_temp->Reshape(bottom[i]->shape());

      const __half* bottom_data = bottom[i]->gpu_data();
      int bottom_data_count = bottom[i]->count();
      Dtype* bottom_data_temp = bottom_temp->mutable_gpu_data();
      convert_to_float<<<CAFFE_GET_BLOCKS(bottom_data_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_data_count, bottom_data, bottom_data_temp);
      const Dtype* bottom_data_converted = bottom_temp->gpu_data();

      Dtype* bottom_diff_temp = bottom_temp->mutable_gpu_diff();

      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data_converted + n * this->bottom_dim_,
              top_diff_converted + n * this->top_dim_, weight_diff_temp);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff_converted + n * this->top_dim_, weight_converted,
              bottom_diff_temp + n * this->bottom_dim_);
        }
      }

      __half* bottom_diff = bottom[i]->mutable_gpu_diff();
      int bottom_diff_count = bottom[i]->count();
      convert_to_fp16<<<CAFFE_GET_BLOCKS(bottom_diff_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_diff_count, bottom_diff_temp, bottom_diff);

    }
  }
  __half* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  int weight_diff_count = this->blobs_[0]->count();
  convert_to_fp16<<<CAFFE_GET_BLOCKS(weight_diff_count), CAFFE_CUDA_NUM_THREADS>>>(weight_diff_count, weight_diff_temp, weight_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
