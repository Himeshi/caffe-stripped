#include "caffe/fp16.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

__global__ void convert_to_fp16(const int n, float* in, fp16* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = fp32tofp16_gpu(in[index]);
  }
}

__global__ void convert_to_fp16(const int n, double* in, fp16* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp32tofp16_gpu(in[index]);
  }
}
__global__ void convert_to_float(const int n,  fp16* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
}

__global__ void convert_to_float(const int n,  fp16* in, double* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
}

__global__ void convert_to_float(const int n, const fp16* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
}

__global__ void convert_to_float(const int n, const fp16* in, double* out) {
  CUDA_KERNEL_LOOP(index, n) {
   out[index] = fp16tofp32_gpu(in[index]);
  }
  
}

__global__ void outputweights(const int n, float* in) {
  CUDA_KERNEL_LOOP(index, n) {
printf("%d %f\n", index, in[index]);
  }
}

__global__ void outputweights(const int n, double* in) {
  CUDA_KERNEL_LOOP(index, n) {
printf("%d %f\n", index, in[index]);
  }
}

void print_gpu_float_array(const float* d_data, int size) {
	float *h_data;
	h_data = (float *) malloc(size * sizeof(float));
	cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
	printf("size = %d\n", size);
	int i;
	for (i = 0; i < size; i++) {
	    if(h_data[i] != 0.0)
		  printf("data[%d] = %f ", i, h_data[i]);
	}
	free(h_data);
}

void print_gpu_float_array(const double* d_data, int size) {
	double *h_data;
	h_data = (double *) malloc(size * sizeof(double));
	cudaMemcpy(h_data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);
	printf("size = %d\n", size);
	int i;
	for (i = 0; i < size; i++) {
	    if(h_data[i] != 0.0)
		  printf("data[%d] = %f ", i, h_data[i]);
	}
	free(h_data);
}

void print_gpu_fp16_array(const fp16* d_data, int size) {
	fp16 *h_data;
	h_data = (fp16 *) malloc(size * sizeof(fp16));
	cudaMemcpy(h_data, d_data, size * sizeof(fp16), cudaMemcpyDeviceToHost);
	printf("size = %d\n", size);
	int i;
	for (i = 0; i < size; i++) {
	    if(h_data[i] != 0)
		  printf("data[%d] = %f ", i, fp16tofp32(h_data[i]));
	}
	free(h_data);
}

}
