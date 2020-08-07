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
  caffe_gpu_add(count_, top[0]->gpu_diff(), top[1]->gpu_diff(),
                bottom[0]->mutable_gpu_diff());
  // Add remaining top blob diffs.
  for (int i = 2; i < top.size(); ++i) {
    const fp16* top_diff = top[i]->gpu_diff();
    fp16* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_axpy_half(count_, Dtype(1.), top_diff, bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);

}  // namespace caffe
