#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<fp16>*>& bottom, const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& top_dtype) {

  /*const fp16* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
  int weight_count = this->blobs_[0]->count();
  convert_to_float<<<CAFFE_GET_BLOCKS(weight_count), CAFFE_CUDA_NUM_THREADS>>>(weight_count, weight, weight_temp);
  const Dtype* weight_temp_data = this->blobs_dtype_[0]->gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const fp16* bottom_data = bottom[i]->gpu_data();
    Blob<Dtype>* temp_bottom = (this->temp_bottom_);
    temp_bottom->Reshape(bottom[i]->shape());
    Dtype* temp_bottom_converted = temp_bottom->mutable_gpu_data();
    int bottom_count = bottom[i]->count();
    convert_to_float<<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_count, bottom_data, temp_bottom_converted);
    const Dtype* temp_bottom_data = temp_bottom->gpu_data();

    Blob<Dtype>* top_temp = (this->temp_top_);
    top_temp->Reshape(top[0]->shape());
    Dtype* top_data_temp = top_temp->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], temp_bottom_data + bottom_offset_ * g,
            filter_desc_, weight_temp_data + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data_temp + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const fp16* bias_data = this->blobs_[1]->gpu_data();
        Dtype* bias_temp = this->blobs_dtype_[1]->mutable_gpu_data();
        int bias_count = this->blobs_[1]->count();
        convert_to_float<<<CAFFE_GET_BLOCKS(bias_count), CAFFE_CUDA_NUM_THREADS>>>(bias_count, bias_data, bias_temp);
        const Dtype* bias_temp_data = this->blobs_dtype_[1]->gpu_data();

        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_temp_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data_temp + top_offset_ * g));
      }
    }

    fp16* top_data = top[i]->mutable_gpu_data();
    int top_data_count = top[i]->count();
    convert_to_fp16<<<CAFFE_GET_BLOCKS(top_data_count), CAFFE_CUDA_NUM_THREADS>>>(top_data_count, top_data_temp, top_data);

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }*/
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom) {
  /*const Dtype* weight_temp_data = NULL;
  Dtype* weight_diff_temp = NULL;
  if (this->param_propagate_down_[0]) {
    const fp16* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
    int weight_count = this->blobs_[0]->count();
    convert_to_float<<<CAFFE_GET_BLOCKS(weight_count), CAFFE_CUDA_NUM_THREADS>>>(weight_count, weight, weight_temp);
    weight_temp_data = this->blobs_dtype_[0]->gpu_data();

    weight_diff_temp = this->blobs_dtype_[0]->mutable_gpu_diff();
  }

  Dtype* bias_diff_temp = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff_temp = this->blobs_dtype_[1]->mutable_gpu_diff();
  }

  for (int i = 0; i < top.size(); ++i) {
    const fp16* top_diff = top[i]->gpu_diff();
    Blob<Dtype>* temp_top = (this->temp_top_);
    temp_top->Reshape(top[i]->shape());
    Dtype* temp_top_converted = temp_top->mutable_gpu_diff();
    int top_count = top[i]->count();
    convert_to_float<<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(top_count, top_diff, temp_top_converted);
    const Dtype* temp_top_diff = temp_top->gpu_diff();

    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  temp_top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff_temp + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const fp16* bottom_data = bottom[i]->gpu_data();
        Blob<Dtype>* temp_bottom = (this->temp_bottom_);
        temp_bottom->Reshape(bottom[i]->shape());
        Dtype* temp_bottom_converted = temp_bottom->mutable_gpu_data();
        int bottom_count = bottom[i]->count();
        convert_to_float<<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_count, bottom_data, temp_bottom_converted);
        const Dtype* temp_bottom_data = temp_bottom->gpu_data();

        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], temp_bottom_data + bottom_offset_ * g,
              top_descs_[i],    temp_top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff_temp + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight_diff_temp == NULL) {
          const fp16* weight = this->blobs_[0]->gpu_data();
          Dtype* weight_temp = this->blobs_dtype_[0]->mutable_gpu_data();
          int weight_count = this->blobs_[0]->count();
          convert_to_float<<<CAFFE_GET_BLOCKS(weight_count), CAFFE_CUDA_NUM_THREADS>>>(weight_count, weight, weight_temp);
          weight_temp_data = this->blobs_dtype_[0]->gpu_data();
        }

        Blob<Dtype>* temp_bottom = (this->temp_bottom_);
        temp_bottom->Reshape(bottom[i]->shape());
        Dtype* temp_bottom_diff = temp_bottom->mutable_gpu_diff();

        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_temp_data + this->weight_offset_ * g,
              top_descs_[i], temp_top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], temp_bottom_diff + bottom_offset_ * g));

        fp16* bottom_diff = bottom[i]->mutable_gpu_diff();
        int bottom_diff_count = bottom[i]->count();
        convert_to_fp16<<<CAFFE_GET_BLOCKS(bottom_diff_count), CAFFE_CUDA_NUM_THREADS>>>(bottom_diff_count, temp_bottom_diff, bottom_diff);
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }

  if (this->param_propagate_down_[0]) {
    fp16* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    int weight_diff_count = this->blobs_[0]->count();
    convert_to_fp16<<<CAFFE_GET_BLOCKS(weight_diff_count), CAFFE_CUDA_NUM_THREADS>>>(weight_diff_count, weight_diff_temp, weight_diff);
  }

  if (this->bias_term_ && this->param_propagate_down_[1]) {
    fp16* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    int bias_diff_count = this->blobs_[1]->count();
    convert_to_fp16<<<CAFFE_GET_BLOCKS(bias_diff_count), CAFFE_CUDA_NUM_THREADS>>>(bias_diff_count, bias_diff_temp, bias_diff);
  }*/
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
