#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<fp16>*>& bottom,
      const vector<Blob<fp16>*>& top, const vector<Blob<Dtype>*>& bottom_dtype,
      const vector<Blob<Dtype>*>& top_dtype) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
    top[i]->data_bias = bottom[i]->data_bias;
    top[i]->diff_bias = bottom[i]->diff_bias;
    if(this->layer_param_.name() == "label_mnist_1_split") {
      top_dtype[i]->Reshape(bottom_dtype[0]->shape());
      top_dtype[i]->ShareData(*bottom_dtype[0]);
    }
  }
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<fp16>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<fp16>*>& bottom,
	  const vector<Blob<Dtype>*>& top_dtype, const vector<Blob<Dtype>*>& bottom_dtype) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    return;
  }
  this->temp_bottom_->Reshape(bottom[0]->shape());
  Dtype* temp_bottom_diff = this->temp_bottom_->mutable_gpu_diff();

  this->temp_top_->Reshape(top[0]->shape());
  Dtype* temp_top_diff_0 = this->temp_top_->mutable_gpu_data();
  caffe_expand_blob_ag(count_, temp_top_diff_0, top[0]->gpu_diff(), top[0]->diff_bias);

  Dtype* temp_top_diff_1 = this->temp_top_->mutable_gpu_diff();
  caffe_expand_blob_ag(count_, temp_top_diff_1, top[1]->gpu_diff(), top[1]->diff_bias);

  caffe_gpu_add(count_, temp_top_diff_0, temp_top_diff_1,
		  temp_bottom_diff);
  caffe_compress_blob_ag(count_, temp_bottom_diff, bottom[0]->mutable_gpu_diff(), &(bottom[0]->diff_bias));

  // Add remaining top blob diffs.
  for (int i = 2; i < top.size(); ++i) {
    const fp16* top_diff = top[i]->gpu_diff();
    this->temp_top_->Reshape(top[0]->shape());
    Dtype* temp_top_diff = this->temp_top_->mutable_gpu_diff();
    caffe_expand_blob_ag(count_, temp_top_diff, top[i]->gpu_diff(), top[i]->diff_bias);

    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    this->temp_bottom_->Reshape(bottom[0]->shape());
    Dtype* temp_bottom_diff = this->temp_bottom_->mutable_gpu_diff();

    caffe_gpu_axpy(count_, Dtype(1.), temp_top_diff, temp_bottom_diff);
    caffe_compress_blob_ag(count_, temp_bottom_diff, bottom[0]->mutable_gpu_diff(), &(bottom[0]->diff_bias));
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);

}  // namespace caffe
