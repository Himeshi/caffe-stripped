#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.cuh"
namespace caffe {

template <>
void caffe_gpu_gemm<fp16>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const fp16 alpha, const fp16* A, const fp16* B, const fp16 beta,
    fp16* C) {
  float* tempA;
  float* tempB;
  float* tempC;

  float tempAlpha = fp16tofp32(alpha);
  float tempBeta = fp16tofp32(beta);

  cudaMalloc(&tempA, K * M * sizeof(float));
  cudaMalloc(&tempB, N * K * sizeof(float));
  cudaMalloc(&tempC, M * N * sizeof(float));

  convert_to_float<<<CAFFE_GET_BLOCKS(K * M), CAFFE_CUDA_NUM_THREADS>>>(K * M, A, tempA);
  convert_to_float<<<CAFFE_GET_BLOCKS(N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);

  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &tempAlpha, tempB, ldb, tempA, lda, &tempBeta, tempC, N));

  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
  cudaFree(tempA);
  cudaFree(tempB);
  cudaFree(tempC);
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm_half<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const fp16* A, const fp16* B, const float beta,
    fp16* C) {
  float* tempA;
  float* tempB;
  float* tempC;
 
  cudaMalloc(&tempA, K * M * sizeof(float));
  cudaMalloc(&tempB, N * K * sizeof(float));
  cudaMalloc(&tempC, M * N * sizeof(float));

  convert_to_float<<<CAFFE_GET_BLOCKS(K * M), CAFFE_CUDA_NUM_THREADS>>>(K * M, A, tempA);
  convert_to_float<<<CAFFE_GET_BLOCKS(N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);

  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, tempB, ldb, tempA, lda, &beta, tempC, N));

  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
  cudaFree(tempA);
  cudaFree(tempB);
  cudaFree(tempC);
}

template <>
void caffe_gpu_gemm_half<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const fp16* A, const fp16* B, const double beta,
    fp16* C) {
  double* tempA;
  double* tempB;
  double* tempC;
  cudaMalloc(&tempA, K * M * sizeof(double));
  cudaMalloc(&tempB, N * K * sizeof(double));
  cudaMalloc(&tempC, M * N * sizeof(double));

  convert_to_float<<<CAFFE_GET_BLOCKS(K * N), CAFFE_CUDA_NUM_THREADS>>>(K * M, A, tempA);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);

  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, tempB, ldb, tempA, lda, &beta, tempC, N));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
  
  cudaFree(tempA);
  cudaFree(tempB);
  cudaFree(tempC);
}

template <>
void caffe_gpu_gemm_half2<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const fp16* B, const float beta,
    fp16* C) {
  float* tempB;
  float* tempC;
  cudaMalloc(&tempB, M * K * sizeof(float));
  cudaMalloc(&tempC, M * N * sizeof(float));

  convert_to_float<<<CAFFE_GET_BLOCKS(M * K), CAFFE_CUDA_NUM_THREADS>>>(M * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);

  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, tempB, ldb, A, lda, &beta, tempC, N));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
  cudaFree(tempB);
  cudaFree(tempC);
}

template <>
void caffe_gpu_gemm_half2<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const fp16* B, const double beta,
    fp16* C) {
  double* tempB;
  double* tempC;
  cudaMalloc(&tempB, M * K * sizeof(double));
  cudaMalloc(&tempC, M * N * sizeof(double));

  convert_to_float<<<CAFFE_GET_BLOCKS(M * K), CAFFE_CUDA_NUM_THREADS>>>(M * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);

  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, tempB, ldb, A, lda, &beta, tempC, N));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
  cudaFree(tempB);
  cudaFree(tempC);
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<fp16>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const fp16 alpha, const fp16* A, const fp16* x,
    const fp16 beta, fp16* y) {
  float* tempA;
  float* tempX;
  float* tempY;
  int asize = M * N;
  cudaMalloc(&tempA, asize * sizeof(float));
  float tempAlpha = fp16tofp32(alpha);
  float tempBeta = fp16tofp32(beta);

  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

  if(cuTransA) {
    cudaMalloc(&tempY, M * sizeof(float));
    cudaMalloc(&tempX, N * sizeof(float));

    convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, x, tempX);
    convert_to_float<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, y, tempY);
  } else {
    cudaMalloc(&tempY, N * sizeof(float));
    cudaMalloc(&tempX, M * sizeof(float));

    convert_to_float<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, x, tempX);
    convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, y, tempY);
  }

  convert_to_float<<<CAFFE_GET_BLOCKS(asize), CAFFE_CUDA_NUM_THREADS>>>(asize, A, tempA);

  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &tempAlpha,
      tempA, N, tempX, 1, &tempBeta, tempY, 1));
  if(cuTransA) {
    convert_to_fp16<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, tempY, y);
  } else {
    convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, y);
  }
  cudaFree(tempA);
  cudaFree(tempX);
  cudaFree(tempY);
}

template <>
void caffe_gpu_gemv_half<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const fp16* A, const float* x,
    const float beta, fp16* y) {
  float* tempA;
  float* tempY;
  int asize = M * N;
  cudaMalloc(&tempA, asize * sizeof(float));
  cudaMalloc(&tempY, N * sizeof(float));

  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  if(cuTransA) {
    cudaMalloc(&tempY, M * sizeof(float));
    convert_to_float<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, y, tempY);
  } else {
    cudaMalloc(&tempY, N * sizeof(float));
    convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, y, tempY);
  }

  convert_to_float<<<CAFFE_GET_BLOCKS(asize), CAFFE_CUDA_NUM_THREADS>>>(asize, A, tempA);

  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha, tempA, N, x, 1, &beta, tempY, 1));
  if(cuTransA) {
    convert_to_fp16<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, tempY, y);
  } else {
    convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, y);
  }
  cudaFree(tempA);
  cudaFree(tempY);
}

template <>
void caffe_gpu_gemv_half<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const fp16* A, const double* x,
    const double beta, fp16* y) {
  double* tempA;
  double* tempY;
  int asize = M * N;
  cudaMalloc(&tempA, asize * sizeof(float));
  cudaMalloc(&tempY, N * sizeof(float));

  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  if(cuTransA) {
    cudaMalloc(&tempY, M * sizeof(float));
    convert_to_float<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, y, tempY);
  } else {
    cudaMalloc(&tempY, N * sizeof(float));
    convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, y, tempY);
  }

  convert_to_float<<<CAFFE_GET_BLOCKS(asize), CAFFE_CUDA_NUM_THREADS>>>(asize, A, tempA);

  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      tempA, N, x, 1, &beta, tempY, 1));
  if(cuTransA) {
    convert_to_fp16<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, tempY, y);
  } else {
    convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, y);
  }
  cudaFree(tempA);
  cudaFree(tempY);
}

void caffe_gpu_axpy_half(const int N, const float alpha, const fp16* X,
    fp16* Y) {
  float* tempX;
  float* tempY;
  cudaMalloc(&tempX, N * sizeof(float));
  cudaMalloc(&tempY, N * sizeof(float)); 
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX);
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, tempX, 1, tempY, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, Y);
  cudaFree(tempX);
  cudaFree(tempY);
}

void caffe_gpu_axpy_half(const int N, const double alpha, const fp16* X,
    fp16* Y) {
  double* tempX;
  double* tempY;
  cudaMalloc(&tempX, N * sizeof(double));
  cudaMalloc(&tempY, N * sizeof(double));   
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX);
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, tempX, 1, tempY, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, Y);
  cudaFree(tempX);
  cudaFree(tempY);
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<fp16>(const int N, const fp16 alpha, const fp16* X,
    fp16* Y) {
  float* tempX;
  float* tempY;
  cudaMalloc(&tempX, N * sizeof(float));
  cudaMalloc(&tempY, N * sizeof(float)); 
  const float alpha_float = fp16tofp32(alpha);

  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX);
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, Y, tempY);

  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha_float, tempX, 1, tempY, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, Y);
  cudaFree(tempX);
  cudaFree(tempY);
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

void caffe_gpu_scal_half(const int N, const float alpha, fp16 *X) {
  float* tempX;
  cudaMallocManaged(&tempX, N*sizeof(float));
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX);
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, tempX, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempX, X);
  cudaFree(tempX);
}

void caffe_gpu_scal_half(const int N, const double alpha, fp16 *X) {
  double* tempX;
  cudaMallocManaged(&tempX, N*sizeof(double));
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX);
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, tempX, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempX, X);
  cudaFree(tempX);
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}
template <>
void caffe_gpu_scal<fp16>(const int N, const fp16 alpha, fp16 *X) {
  float* tempX;
  cudaMallocManaged(&tempX, N*sizeof(float));
  const float alpha_float = fp16tofp32(alpha);
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX);
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha_float, tempX, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempX, X);
  cudaFree(tempX);
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

void caffe_gpu_axpby_half(const int N, const float alpha, const fp16* X,
    const float beta, fp16* Y) {
  caffe_gpu_scal_half(N, beta, Y);
  caffe_gpu_axpy_half(N, alpha, X, Y);
}

void caffe_gpu_axpby_half(const int N, const double alpha, const fp16* X,
    const double beta, fp16* Y) {
  caffe_gpu_scal_half(N, beta, Y);
  caffe_gpu_axpy_half(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot_half<float>(const int n, const fp16* x, const fp16* y,
    float* out) {
  float* tempX;
  float* tempY;
  cudaMalloc(&tempX, n * sizeof(float));
  cudaMalloc(&tempY, n * sizeof(float));
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, y, tempY);
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, tempX, 1, tempY, 1, out));
  cudaFree(tempX);
  cudaFree(tempY);
}

template <>
void caffe_gpu_dot_half<double>(const int n, const fp16* x, const fp16* y,
    double* out) {
  double* tempX;
  double* tempY;
  cudaMalloc(&tempX, n * sizeof(double));
  cudaMalloc(&tempY, n * sizeof(double));
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, y, tempY);
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, tempX, 1, tempY, 1, out));
  cudaFree(tempX);
  cudaFree(tempY);
}


template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<fp16>(const int n, const fp16* x, const fp16* y,
    fp16 * out) {
  float* tempX;
  cudaMallocManaged(&tempX, n*sizeof(float));
  float* tempY;
  cudaMallocManaged(&tempY, n*sizeof(float));
  float* tempOut;
  cudaMallocManaged(&tempOut, n*sizeof(float));
  
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, y, tempY);
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, tempX, 1, tempY, 1, tempOut));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, tempOut, out);

  cudaFree(tempX);
  cudaFree(tempY);
  cudaFree(tempOut);
}

template <>
void caffe_gpu_asum_half<float>(const int n, const fp16* x, float* y) {
  float* tempX;
  cudaMalloc(&tempX, n * sizeof(float));
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, tempX, 1, y));
  cudaFree(tempX);
}

template <>
void caffe_gpu_asum_half<double>(const int n, const fp16* x, double* y) {
  double* tempX;
  cudaMalloc(&tempX, n * sizeof(double));
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, tempX, 1, y));
  cudaFree(tempX);
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<fp16>(const int n, const fp16* x, fp16* y) {
  float tempY;
  float* tempX;
  cudaMallocManaged(&tempX, n*sizeof(float)); 
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, tempX, 1, &tempY));
  *y = fp32tofp16(tempY);
  cudaFree(tempX);
}

void caffe_gpu_scale_half(const int n, const float alpha, const fp16 *x,
                            fp16* y) {
  float* tempX;
  cudaMallocManaged(&tempX, n*sizeof(float));
  float* tempY;
  cudaMallocManaged(&tempY, n*sizeof(float));
   
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, tempX, 1, tempY, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, tempY, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, tempY, y);

  cudaFree(tempX);
  cudaFree(tempY);
}

void caffe_gpu_scale_half(const int n, const double alpha, const fp16 *x,
                            fp16* y) {

  double* tempX;
  cudaMallocManaged(&tempX, n*sizeof(double));
  double* tempY;
  cudaMallocManaged(&tempY, n*sizeof(double));
    
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, tempX, 1, tempY, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, tempY, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, tempY, y);
  
  cudaFree(tempX);
  cudaFree(tempY);
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

void caffe_gpu_set_half(const int N, const float alpha, fp16* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(fp16) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  fp16 alpha_temp = fp32tofp16(alpha);
  set_kernel<fp16><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha_temp, Y);
}

void caffe_gpu_set_half(const int N, const double alpha, fp16* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(fp16) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  fp16 alpha_temp = fp32tofp16(alpha);
  set_kernel<fp16><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha_temp, Y);
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

__global__ void add_scalar_kernel_half(const int n, const float alpha, fp16* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fp32tofp16_gpu(fp16tofp32_gpu(y[index]) + alpha);
  }
}

__global__ void add_scalar_kernel_half(const int n, const double alpha, fp16* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fp32tofp16_gpu(fp16tofp32_gpu(y[index]) + alpha);
  }
}

void caffe_gpu_add_scalar_half(const int N, const float alpha, fp16* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel_half<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

void caffe_gpu_add_scalar_half(const int N, const double alpha, fp16* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel_half<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

__global__ void add_kernel_half(const int n, const fp16* a,
    const fp16* b, fp16* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fp32tofp16_gpu(fp16tofp32_gpu(a[index]) + fp16tofp32_gpu(b[index]));
  }
}

template <>
void caffe_gpu_add<fp16>(const int N, const fp16* a, const fp16* b,
    fp16* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel_half<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

__global__ void mul_kernel_half(const int n, const fp16* a,
    const fp16* b, fp16* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fp32tofp16_gpu(fp16tofp32_gpu(a[index]) * fp16tofp32_gpu(b[index]));
  }
}

template <>
void caffe_gpu_mul<fp16>(const int N, const fp16* a,
    const fp16* b, fp16* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel_half<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

__global__ void div_kernel_half(const int n, const fp16* a,
    const fp16* b, fp16* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fp32tofp16_gpu(fp16tofp32_gpu(a[index]) / fp16tofp32_gpu(b[index]));
  }
}

template <>
void caffe_gpu_div<fp16>(const int N, const fp16* a,
    const fp16* b, fp16* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel_half<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

__global__ void powx_kernel_half(const int n, const fp16* a,
    const float alpha, fp16* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fp32tofp16_gpu(pow(fp16tofp32_gpu(a[index]), alpha));
  }
}

__global__ void powx_kernel_half(const int n, const fp16* a,
    const double alpha, fp16* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fp32tofp16_gpu(pow(fp16tofp32_gpu(a[index]), alpha));
  }
}

void caffe_gpu_powx_half(const int N, const fp16* a,
    const float alpha, fp16* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel_half<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

void caffe_gpu_powx_half(const int N, const fp16* a,
    const double alpha, fp16* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel_half<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));

__global__ void sign_kernel_half(const int n, const fp16* x, fp16* y) {
  CUDA_KERNEL_LOOP(index, n) {
    short x_index = x[index];
    y[index] = (fp16(0) < x_index) - (x_index < fp16(0));
  }
}

template <>
void caffe_gpu_sign<fp16>(const int n, const fp16* x, fp16* y) {
  /* NOLINT_NEXT_LINE(whitespace/operators) */
  sign_kernel_half<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
