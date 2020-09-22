#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fp16.hpp"
#include "caffe/fp16.cuh"

#define CUSTOM_MAT_MUL
//#define GEMM_NO_MALLOC
// Thread block size
#define BLOCK_SIZE 11

#define CUSTOM_GEMV
//gemv block size must always be an even number!
#define GEMV_BLOCK_SIZE 16

#define CUSTOM_ASUM
#define ASUM_BLOCK_SIZE 16

#define CUSTOM_AXPY
#define AXPY_BLOCK_SIZE 16

#define CUSTOM_DOT
#define DOT_BLOCK_SIZE 16

#define SCAL_BLOCK_SIZE 16


namespace caffe {

__global__ void MatMulSharedMemKernel(const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const float alpha, const fp16* A, const int lda, const fp16* B, const float beta,
    const int ldb, fp16* C) {

  // Block row and column
  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;

  // Thread row and column
  int row = threadIdx.x;
  int col = threadIdx.y;

  int cRow = blockRow * BLOCK_SIZE + row;
  int cCol = blockCol * BLOCK_SIZE + col;
  int cIndex = cRow * M + cCol;

  float cValue = 0;

  int iter = ((K + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int count = BLOCK_SIZE;

  int aRow, aCol, bRow, bCol, aIndex, bIndex;
  for (int m = 0; m < iter; ++m) {

    if(m == iter - 1)
      count = K - (m * BLOCK_SIZE);

    // Shared memory used to store submatrices
    __shared__ fp16 As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ fp16 Bs[BLOCK_SIZE][BLOCK_SIZE];

    if(TransA == 0) {
      aRow = m * BLOCK_SIZE + row;
      aCol = cCol;
      aIndex = aRow * M + aCol;
      if(aRow < K && aCol < M)
        As[col][row] = A[aIndex];
    } else {
      aRow = cCol;
      aCol = m * BLOCK_SIZE + row;
      aIndex = aRow * K + aCol;
      if(aRow < M && aCol < K)
        As[col][row] = A[aIndex];
    }

    if(TransB == 0) {
      bRow = cRow;
      bCol = m * BLOCK_SIZE + col;
      bIndex = bRow * K + bCol;
      if(bRow < N && bCol < K)
        Bs[row][col] = B[bIndex];
    } else {
      bRow = m * BLOCK_SIZE + col;
      bCol = cRow;
      bIndex = bRow * N + bCol;
      if(bRow < K && bCol < N)
        Bs[row][col] = B[bIndex];
    }

    __syncthreads();

    for (int e = 0; e < count; ++e) {
      cValue += fp16tofp32_gpu(As[col][e]) * alpha * fp16tofp32_gpu(Bs[row][e]);
    }

    __syncthreads();
  }

  if(cRow < N && cCol < M)
    C[cIndex] = fp32tofp16_gpu(cValue + (beta * fp16tofp32_gpu(C[cIndex])));
}

__global__ void MatMulSharedMemKernelWithFloat(const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const float alpha, const fp16* A, const int lda, const float* B, const float beta,
    const int ldb, fp16* C) {

  // Block row and column
  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;

  // Thread row and column
  int row = threadIdx.x;
  int col = threadIdx.y;

  int cRow = blockRow * BLOCK_SIZE + row;
  int cCol = blockCol * BLOCK_SIZE + col;
  int cIndex = cRow * M + cCol;

  float cValue = 0;

  int iter = ((K + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int count = BLOCK_SIZE;

  int aRow, aCol, bRow, bCol, aIndex, bIndex;
  for (int m = 0; m < iter; ++m) {

    if(m == iter - 1)
      count = K - (m * BLOCK_SIZE);

    // Shared memory used to store submatrices
    __shared__ fp16 As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    if(TransA == 0) {
      aRow = m * BLOCK_SIZE + row;
      aCol = cCol;
      aIndex = aRow * M + aCol;
      if(aRow < K && aCol < M)
        As[col][row] = A[aIndex];
    } else {
      aRow = cCol;
      aCol = m * BLOCK_SIZE + row;
      aIndex = aRow * K + aCol;
      if(aRow < M && aCol < K)
        As[col][row] = A[aIndex];
    }

    if(TransB == 0) {
      bRow = cRow;
      bCol = m * BLOCK_SIZE + col;
      bIndex = bRow * K + bCol;
      if(bRow < N && bCol < K)
        Bs[row][col] = B[bIndex];
    } else {
      bRow = m * BLOCK_SIZE + col;
      bCol = cRow;
      bIndex = bRow * N + bCol;
      if(bRow < K && bCol < N)
        Bs[row][col] = B[bIndex];
    }

    __syncthreads();

    for (int e = 0; e < count; ++e) {
      cValue += fp16tofp32_gpu(As[col][e]) * alpha * Bs[row][e];
    }

    __syncthreads();
  }

  if(cRow < N && cCol < M)
    C[cIndex] = fp32tofp16_gpu(cValue + (beta * fp16tofp32_gpu(C[cIndex])));
}

__global__ void MatMulSharedMemKernelWithFloatB(const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const int lda, const fp16* B, const float beta,
    const int ldb, fp16* C) {

  // Block row and column
  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;

  // Thread row and column
  int row = threadIdx.x;
  int col = threadIdx.y;

  int cRow = blockRow * BLOCK_SIZE + row;
  int cCol = blockCol * BLOCK_SIZE + col;
  int cIndex = cRow * M + cCol;

  float cValue = 0;

  int iter = ((K + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int count = BLOCK_SIZE;

  int aRow, aCol, bRow, bCol, aIndex, bIndex;
  for (int m = 0; m < iter; ++m) {

    if(m == iter - 1)
      count = K - (m * BLOCK_SIZE);

    // Shared memory used to store submatrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ fp16 Bs[BLOCK_SIZE][BLOCK_SIZE];

    if(TransA == 0) {
      aRow = m * BLOCK_SIZE + row;
      aCol = cCol;
      aIndex = aRow * M + aCol;
      if(aRow < K && aCol < M)
        As[col][row] = A[aIndex];
    } else {
      aRow = cCol;
      aCol = m * BLOCK_SIZE + row;
      aIndex = aRow * K + aCol;
      if(aRow < M && aCol < K)
        As[col][row] = A[aIndex];
    }

    if(TransB == 0) {
      bRow = cRow;
      bCol = m * BLOCK_SIZE + col;
      bIndex = bRow * K + bCol;
      if(bRow < N && bCol < K)
        Bs[row][col] = B[bIndex];
    } else {
      bRow = m * BLOCK_SIZE + col;
      bCol = cRow;
      bIndex = bRow * N + bCol;
      if(bRow < K && bCol < N)
        Bs[row][col] = B[bIndex];
    }

    __syncthreads();

    for (int e = 0; e < count; ++e) {
      cValue += As[col][e] * alpha * fp16tofp32_gpu(Bs[row][e]);
    }

    __syncthreads();
  }

  if(cRow < N && cCol < M)
    C[cIndex] = fp32tofp16_gpu(cValue + (beta * fp16tofp32_gpu(C[cIndex])));
}

#define CUDA_BUFFER_SIZE 1000000
float* cuda_buffer; //device buffer
//device buffer size
void init_cuda_buffer() {
#ifdef GEMM_NO_MALLOC
  cudaMalloc(&cuda_buffer, CUDA_BUFFER_SIZE * sizeof(float));
#endif
}
__global__ void MatMulSharedMemKernelWithFloatInputs(const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const int lda, const float* B, const float beta,
    const int ldb, fp16* C) {

  // Block row and column
  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;

  // Thread row and column
  int row = threadIdx.x;
  int col = threadIdx.y;

  int cRow = blockRow * BLOCK_SIZE + row;
  int cCol = blockCol * BLOCK_SIZE + col;
  int cIndex = cRow * M + cCol;

  float cValue = 0;

  int iter = ((K + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int count = BLOCK_SIZE;

  int aRow, aCol, bRow, bCol, aIndex, bIndex;
  for (int m = 0; m < iter; ++m) {

    if(m == iter - 1)
      count = K - (m * BLOCK_SIZE);

    // Shared memory used to store submatrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    if(TransA == 0) {
      aRow = m * BLOCK_SIZE + row;
      aCol = cCol;
      aIndex = aRow * M + aCol;
      if(aRow < K && aCol < M)
        As[col][row] = A[aIndex];
    } else {
      aRow = cCol;
      aCol = m * BLOCK_SIZE + row;
      aIndex = aRow * K + aCol;
      if(aRow < M && aCol < K)
        As[col][row] = A[aIndex];
    }

    if(TransB == 0) {
      bRow = cRow;
      bCol = m * BLOCK_SIZE + col;
      bIndex = bRow * K + bCol;
      if(bRow < N && bCol < K)
        Bs[row][col] = B[bIndex];
    } else {
      bRow = m * BLOCK_SIZE + col;
      bCol = cRow;
      bIndex = bRow * N + bCol;
      if(bRow < K && bCol < N)
        Bs[row][col] = B[bIndex];
    }

    __syncthreads();

    for (int e = 0; e < count; ++e) {
      cValue += As[col][e] * alpha * Bs[row][e];
    }

    __syncthreads();
  }

  if(cRow < N && cCol < M)
    C[cIndex] = fp32tofp16_gpu(cValue + (beta * fp16tofp32_gpu(C[cIndex])));
}

template <>
void caffe_gpu_gemm<fp16>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const fp16 alpha, const fp16* A, const fp16* B, const fp16 beta,
    fp16* C) {

  float tempAlpha = fp16tofp32(alpha);
  float tempBeta = fp16tofp32(beta);

  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef CUSTOM_MAT_MUL
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((M + BLOCK_SIZE - 1) / dimBlock.x, (N + BLOCK_SIZE - 1) / dimBlock.y);
  MatMulSharedMemKernel<<<dimGrid, dimBlock>>>(cuTransB, cuTransA, N, M, K, tempAlpha, B, ldb, A, tempBeta, lda, C);
#else
#ifdef GEMM_NO_MALLOC

  float* tempA = cuda_buffer ;
  float* tempB = cuda_buffer + K * M ;
  float* tempC = cuda_buffer + K * M  + N * K;
  if (K * M +  N * K + M * N > CUDA_BUFFER_SIZE ) {
    printf(" buffer overflow, need to increase CUDA_BUFFER_SIZE to greater than %d \n",K * M +  N * K + M * N  );
    exit(0);
  }
/*
  convert_to_float<<<CAFFE_GET_BLOCKS(K * M), CAFFE_CUDA_NUM_THREADS>>>(K * M, A, tempA);
  convert_to_float<<<CAFFE_GET_BLOCKS(N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);
*/
  convert_to_float_3in1out<<<CAFFE_GET_BLOCKS(K * M +  N * K + M * N ), CAFFE_CUDA_NUM_THREADS>>>(K * M, N * K,M * N, A,B,C, tempA);

  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &tempAlpha, tempB, ldb, tempA, lda, &tempBeta, tempC, N));

  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
#else
  float* tempA;
  float* tempB;
  float* tempC;

  cudaMalloc(&tempA, K * M * sizeof(float));
  cudaMalloc(&tempB, N * K * sizeof(float));
  cudaMalloc(&tempC, M * N * sizeof(float));
  //printf(" K * M  %d N * K  %d  M * N %d \n", K * M, N * K , M * N );

  convert_to_float<<<CAFFE_GET_BLOCKS(K * M), CAFFE_CUDA_NUM_THREADS>>>(K * M, A, tempA);
  convert_to_float<<<CAFFE_GET_BLOCKS(N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);

  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &tempAlpha, tempB, ldb, tempA, lda, &tempBeta, tempC, N));

  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
  cudaFree(tempA);
  cudaFree(tempB);
  cudaFree(tempC);
#endif
#endif
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
void caffe_gpu_gemm_half_with_float(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const fp16* B, const float beta,
	fp16* C) {

  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
#ifdef CUSTOM_MAT_MUL
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((M + BLOCK_SIZE - 1) / dimBlock.x, (N + BLOCK_SIZE - 1) / dimBlock.y);
  MatMulSharedMemKernelWithFloat<<<dimGrid, dimBlock>>>(cuTransB, cuTransA, N, M, K, alpha, B, ldb, A, beta, lda, C);
#else
#ifdef GEMM_NO_MALLOC
  float* tempB = cuda_buffer ;
  float* tempC = cuda_buffer + N * K ;

  //printf("  N * K  %d  M * N %d \n", N * K , M * N );
  if (N * K + M * N > CUDA_BUFFER_SIZE) {
    printf(" buffer overflow, need to increase CUDA_BUFFER_SIZE to greater than %d \n",N * K + M * N );
    exit(0);
  }
/*
  convert_to_float<<<CAFFE_GET_BLOCKS(N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);
*/
  convert_to_float_2in1out<<<CAFFE_GET_BLOCKS(M * N + N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, M*N, B , C, tempB);

  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, tempB, ldb, A, lda, &beta, tempC, N));

  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);

#else
  float* tempB;
  float* tempC;

  cudaMalloc(&tempB, N * K * sizeof(float));
  cudaMalloc(&tempC, M * N * sizeof(float));

  convert_to_float<<<CAFFE_GET_BLOCKS(N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);

  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, tempB, ldb, A, lda, &beta, tempC, N));

  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
  cudaFree(tempB);
  cudaFree(tempC);
#endif
#endif
}

template <>
void caffe_gpu_gemm_half_with_float(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const fp16* B, const double beta,
	fp16* C) {

  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
#ifdef CUSTOM_MAT_MUL
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((M + BLOCK_SIZE - 1) / dimBlock.x, (N + BLOCK_SIZE - 1) / dimBlock.y);
  MatMulSharedMemKernelWithFloat<<<dimGrid, dimBlock>>>(cuTransB, cuTransA, N, M, K, (float) alpha, B, ldb, (float *) A, (float) beta, lda, C);

#else
  double* tempB;
  double* tempC;
  cudaMalloc(&tempB, N * K * sizeof(double));
  cudaMalloc(&tempC, M * N * sizeof(double));

  convert_to_float<<<CAFFE_GET_BLOCKS(N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);

  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, tempB, ldb, A, lda, &beta, tempC, N));

  convert_to_fp16<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, tempC, C);
  cudaFree(tempB);
  cudaFree(tempC);
#endif

}

template <>
void caffe_gpu_gemm_half<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const fp16* A, const fp16* B, const float beta,
    fp16* C) {
  float* tempA;
  float* tempB;
  float* tempC;
  //printf("caffe_gpu_gemm_half called\n");
#ifdef GEMM_NO_MALLOC
  tempA = cuda_buffer;
  tempB = cuda_buffer + K * M ;
  tempC = cuda_buffer + K * M  + N * K;

  if (K * M +  N * K + M * N > CUDA_BUFFER_SIZE ){
    printf(" buffer overflow, need to increase CUDA_BUFFER_SIZE to greater than %d \n",K * M +  N * K + M * N  );
    exit(0);
  }
  /*
  convert_to_float<<<CAFFE_GET_BLOCKS(K * M), CAFFE_CUDA_NUM_THREADS>>>(K * M, A, tempA);
  convert_to_float<<<CAFFE_GET_BLOCKS(N * K), CAFFE_CUDA_NUM_THREADS>>>(N * K, B, tempB);
  convert_to_float<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS>>>(M * N, C, tempC);
*/
  convert_to_float_3in1out<<<CAFFE_GET_BLOCKS(K * M +  N * K + M * N ), CAFFE_CUDA_NUM_THREADS>>>(K * M, N * K,M * N, A,B,C, tempA);

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

#else
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
#endif
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

__global__ void GemvStepOneKernel(cublasOperation_t cuTransA, int M, int N, const float alpha,
    const fp16* A, const fp16* x, float* output) {
  __shared__ fp16 xdata[GEMV_BLOCK_SIZE];
  __shared__ float sum[GEMV_BLOCK_SIZE][GEMV_BLOCK_SIZE];

  int aRow = blockIdx.x * GEMV_BLOCK_SIZE + threadIdx.x;
  int aCol = blockIdx.y * GEMV_BLOCK_SIZE + threadIdx.y;
  int row = threadIdx.x;
  int col = threadIdx.y;

  if(cuTransA == 0) {
    if(aRow < M) {
      xdata[row] = x[aRow];
      __syncthreads();
    }

    if(aRow < M && aCol < N) {
      sum[row][col] = alpha * fp16tofp32_gpu(A[aRow * N + aCol]) * fp16tofp32_gpu(xdata[row]);
      __syncthreads();

      for(int i = GEMV_BLOCK_SIZE / 2; i > 0; i>>=1) {
        if(row < i) {
          sum[row][col] += sum[row + i][col];
        }
        __syncthreads();
      }

      if(row == 0) {
        output[aCol * gridDim.x + blockIdx.x] = sum[row][col];
      }
    } else {
      sum[row][col] = 0;
    }
  } else {
    if(aCol < N) {
      xdata[col] = x[aCol];
      __syncthreads();
    }

    if(aRow < M && aCol < N) {
      sum[row][col] = alpha * fp16tofp32_gpu(A[aRow * N + aCol]) * fp16tofp32_gpu(xdata[col]);
      __syncthreads();

      for(int i = GEMV_BLOCK_SIZE / 2; i > 0; i>>=1) {
        if(col < i) {
          sum[row][col] += sum[row][col + i];
        }
        __syncthreads();
      }

      if(col == 0) {
        output[aRow * gridDim.y + blockIdx.y] = sum[row][col];
      }
    } else {
      sum[row][col] = 0;
    }
  }
}

__global__ void GemvStepTwoKernel(cublasOperation_t cuTransA, float* output, const float beta, fp16* y, int M, int N, int prevStepBlocks) {
  int rowIndex = threadIdx.x + (blockIdx.x * GEMV_BLOCK_SIZE);
  int startIndex = rowIndex * prevStepBlocks;
  float sum = 0.0;
  if(cuTransA == 0) {
    if(rowIndex < N) {
      for(int i = 0; i < prevStepBlocks; i++) {
        sum += output[startIndex + i];
      }
      y[rowIndex] = fp32tofp16_gpu((beta * fp16tofp32_gpu(y[rowIndex])) + sum);
    }
  } else {
    if(rowIndex < M) {
      for(int i = 0; i < prevStepBlocks; i++) {
        sum += output[startIndex + i];
      }
      y[rowIndex] = fp32tofp16_gpu((beta * fp16tofp32_gpu(y[rowIndex])) + sum);
    }
  }
}

template <>
void caffe_gpu_gemv<fp16>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const fp16 alpha, const fp16* A, const fp16* x,
    const fp16 beta, fp16* y) {
  float tempAlpha = fp16tofp32(alpha);
  float tempBeta = fp16tofp32(beta);

  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

#ifdef CUSTOM_GEMV

  dim3 dimBlock(GEMV_BLOCK_SIZE, GEMV_BLOCK_SIZE);
  float* output;
  dim3 dimGrid((M + GEMV_BLOCK_SIZE - 1) / dimBlock.x, (N + GEMV_BLOCK_SIZE - 1) / dimBlock.y);
  if(cuTransA == 0)
    cudaMalloc(&output, N * ((M + GEMV_BLOCK_SIZE - 1) / GEMV_BLOCK_SIZE) * sizeof(float));
  else
    cudaMalloc(&output, M * ((N + GEMV_BLOCK_SIZE - 1) / GEMV_BLOCK_SIZE) * sizeof(float));

  GemvStepOneKernel<<<dimGrid, dimBlock>>>(cuTransA, M, N, tempAlpha, A, x, output);

  if(cuTransA == 0)
    GemvStepTwoKernel<<<(GEMV_BLOCK_SIZE + N - 1) / GEMV_BLOCK_SIZE, GEMV_BLOCK_SIZE>>>(cuTransA, output, tempBeta, y, M, N, dimGrid.x);
  else
    GemvStepTwoKernel<<<(GEMV_BLOCK_SIZE + M - 1) / GEMV_BLOCK_SIZE, GEMV_BLOCK_SIZE>>>(cuTransA, output, tempBeta, y, M, N, dimGrid.y);
  cudaFree(output);
#else
#ifdef GEMM_NO_MALLOC

    float* tempA = cuda_buffer + M + N;
    //float* tempX;
    //float* tempY;
    if (M * N + M + N > CUDA_BUFFER_SIZE ){
      printf(" buffer overflow, need to increase CUDA_BUFFER_SIZE to greater than %d \n",M * N + M + N  );
      exit(0);
    }

    int asize = M * N;
    int XOFFSET = 0 ;
    if(cuTransA) {
     // tempY = cuda_buffer; //, M * sizeof(float));
     // tempX = cuda_buffer + M ;//, N * sizeof(float));
        XOFFSET = M;
    // convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, x, tempX);
    // convert_to_float<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, y, tempY);
    } else {

     // tempY = cuda_buffer; //, M * sizeof(float));
     // tempX = cuda_buffer + N ;//, N * sizeof(float));
        XOFFSET = N;

      // convert_to_float<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, x, tempX);
      // convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, y, tempY);
    }
    float* tempX = cuda_buffer + XOFFSET;
    float* tempY = cuda_buffer;
    // convert_to_float<<<CAFFE_GET_BLOCKS(asize), CAFFE_CUDA_NUM_THREADS>>>(asize, A, tempA);
    convert_to_float_3in1out<<<CAFFE_GET_BLOCKS(asize + M + N ), CAFFE_CUDA_NUM_THREADS>>>(XOFFSET, M +N -  XOFFSET, asize, y,x,A, tempY);


    CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &tempAlpha,
        tempA, N, tempX, 1, &tempBeta, tempY, 1));
    if(cuTransA) {
      convert_to_fp16<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, tempY, y);
    } else {
      convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, y);
    }


#else

  float* tempA;
  float* tempX;
  float* tempY;
  int asize = M * N;
  cudaMalloc(&tempA, asize * sizeof(float));

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

#endif
#endif
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

__global__ void AxpyKernel(const int N, const float alpha, const fp16* X,
    fp16* Y) {
  int index = threadIdx.x + (blockIdx.x * AXPY_BLOCK_SIZE);
  if(index < N) {
    Y[index] = fp32tofp16_gpu((fp16tofp32_gpu(X[index]) * alpha) + fp16tofp32_gpu(Y[index]));
  }
}

template <>
void caffe_gpu_axpy_half(const int N, const float alpha, const fp16* X,
    fp16* Y) {
#ifdef CUSTOM_AXPY
  AxpyKernel<<<(N + AXPY_BLOCK_SIZE - 1) / AXPY_BLOCK_SIZE, AXPY_BLOCK_SIZE>>>(N, alpha, X, Y);
#else
  float* tempX;
  float* tempY;
  cudaMalloc(&tempX, N * sizeof(float));
  cudaMalloc(&tempY, N * sizeof(float));
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX);
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, Y, tempY);

  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, tempX, 1, tempY, 1));

  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, Y);
  cudaFree(tempX);
  cudaFree(tempY);
#endif
}

template <>
void caffe_gpu_axpy_half(const int N, const double alpha, const fp16* X,
    fp16* Y) {
  double* tempX;
  double* tempY;
  cudaMalloc(&tempX, N * sizeof(double));
  cudaMalloc(&tempY, N * sizeof(double));

  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX);
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, Y, tempY);

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
#ifdef CUSTOM_AXPY
  AxpyKernel<<<(N + AXPY_BLOCK_SIZE - 1) / AXPY_BLOCK_SIZE, AXPY_BLOCK_SIZE>>>(N, fp16tofp32(alpha), X, Y);
#else
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
#endif
}

template <>
void caffe_gpu_axpy_with_bias<float>(const int N, const float alpha, const float* X,
    float* Y, float x_bias, float* y_bias) {
//TODO
}

template <>
void caffe_gpu_axpy_with_bias<double>(const int N, const double alpha, const double* X,
    double* Y, float x_bias, float* y_bias) {
//TODO
}

template <>
void caffe_gpu_axpy_with_bias<fp16>(const int N, const fp16 alpha, const fp16* X,
    fp16* Y, float x_bias, float* y_bias) {
  float* tempX;
  float* tempY;
  cudaMalloc(&tempX, N * sizeof(float));
  cudaMalloc(&tempY, N * sizeof(float));
  const float alpha_float = fp16tofp32(alpha);

  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, tempX, x_bias);
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, Y, tempY, *y_bias);

  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha_float, tempX, 1, tempY, 1));

  int max_index;
  CUBLAS_CHECK(cublasIsamax(Caffe::cublas_handle(), N, tempY, 1, &max_index));
  cudaMemcpy(y_bias, tempY + max_index - 1, sizeof(float), cudaMemcpyDeviceToHost);
  if(*y_bias == 0)
    *y_bias = 1.0;
  else
    *y_bias = fabsf(*y_bias);

  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, tempY, Y, *y_bias);
  cudaFree(tempX);
  cudaFree(tempY);
}

template <>
void caffe_expand_blob(int N, float* out,const fp16* in, float bias) {
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, in, out, bias);
}

template <>
void caffe_expand_blob(int N, double* out,const fp16* in, float bias) {
  convert_to_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, in, out, bias);
}

template <>
void caffe_compress_blob(int N, float* in, fp16* out, float* bias) {
  int max_index;
  CUBLAS_CHECK(cublasIsamax(Caffe::cublas_handle(), N, in, 1, &max_index));
  cudaMemcpy(bias, in + max_index - 1, sizeof(float), cudaMemcpyDeviceToHost);
  *bias = fabsf(*bias);
  if(*bias == 0 || isnan(*bias) || isinf(*bias))
    *bias = 1.0;

  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, in, out, *bias);
}

template <>
void caffe_compress_blob(int N, double* in, fp16* out, float* bias) {
  int max_index;
  CUBLAS_CHECK(cublasIdamax(Caffe::cublas_handle(), N, in, 1, &max_index));
  cudaMemcpy(bias, in + max_index - 1, sizeof(float), cudaMemcpyDeviceToHost);
  *bias = fabsf(*bias);
  if(*bias == 0|| isnan(*bias) || isinf(*bias))
    *bias = 1.0;

  convert_to_fp16<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, in, out, *bias);
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

__global__ void scal_kernel(const int n, const float alpha, fp16* X) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < n)
    X[index] = fp32tofp16_gpu(fp16tofp32_gpu(X[index]) * alpha);
}

template <>
void caffe_gpu_scal_half<float>(const int N, const float alpha, fp16 *X) {
  scal_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, X);
}

template <>
void caffe_gpu_scal_half<double>(const int N, const double alpha, fp16 *X) {
  float alpha_float = alpha;
  scal_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha_float, X);
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}
template <>
void caffe_gpu_scal<fp16>(const int N, const fp16 alpha, fp16 *X) {
  const float alpha_float = fp16tofp32(alpha);
  scal_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha_float, X);
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

__global__ void DotKernel(const int n, const fp16* x, const fp16* y,
    float* temp) {
  int globalIndex = threadIdx.x + (blockIdx.x * DOT_BLOCK_SIZE);
  int localIndex = threadIdx.x;
  __shared__ float product[DOT_BLOCK_SIZE];

  if(localIndex < n) {
    product[localIndex] = fp16tofp32_gpu(x[globalIndex]) * fp16tofp32_gpu(y[globalIndex]);
    __syncthreads();

    for(int i = DOT_BLOCK_SIZE / 2; i > 0; i>>=1) {
      if(localIndex < i) {
        product[localIndex] += product[localIndex + i];
      }
      __syncthreads();
    }

    if(localIndex == 0) {
      temp[blockIdx.x] = product[localIndex];
    }
  } else {
    product[localIndex] = 0;
  }
}

__global__ void DotKernelStepTwo(const int n, float* x) {
  __shared__ float xdata[DOT_BLOCK_SIZE];
  int globalIndex = blockIdx.x * DOT_BLOCK_SIZE + threadIdx.x;
  int localIndex = threadIdx.x;

  if(globalIndex < n) {
    xdata[localIndex] = x[globalIndex];
    __syncthreads();

    for(int i = DOT_BLOCK_SIZE / 2; i > 0; i>>=1) {
      if(localIndex < i) {
        xdata[localIndex] += xdata[localIndex + i];
      }
      __syncthreads();
    }

    if(localIndex == 0) {
      x[blockIdx.x] = xdata[localIndex];
    }

  } else {
    xdata[localIndex] = 0;
  }
}

template <>
void caffe_gpu_dot_half<float>(const int n, const fp16* x, const fp16* y,
    float* out) {
#ifdef CUSTOM_DOT
  #ifdef GEMM_NO_MALLOC
    float* temp = cuda_buffer;
    if (((n + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE)> CUDA_BUFFER_SIZE ){
      printf(" buffer overflow, need to increase CUDA_BUFFER_SIZE to greater than %d \n",((n + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE)  );
      exit(0);
    }
  #else
    float* temp;
    cudaMalloc(&temp, ((n + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE)  * sizeof(float));
  #endif
  DotKernel<<<(n + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE, DOT_BLOCK_SIZE>>>(n, x, y, temp);
  int ntemp = (DOT_BLOCK_SIZE + n - 1) / DOT_BLOCK_SIZE;
  while(ntemp > 1) {
    DotKernelStepTwo<<<(DOT_BLOCK_SIZE + ntemp - 1) / DOT_BLOCK_SIZE, DOT_BLOCK_SIZE>>>(ntemp, temp);
    ntemp = (DOT_BLOCK_SIZE + ntemp - 1) / DOT_BLOCK_SIZE;
  }
  cudaMemcpy(out, temp, sizeof(float), cudaMemcpyDeviceToHost);
  #ifndef GEMM_NO_MALLOC
    cudaFree(temp);
  #endif
#else
  float* tempX;
  float* tempY;
  cudaMalloc(&tempX, n * sizeof(float));
  cudaMalloc(&tempY, n * sizeof(float));
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, y, tempY);
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, tempX, 1, tempY, 1, out));
  cudaFree(tempX);
  cudaFree(tempY);
#endif
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
  cudaMalloc(&tempX, n*sizeof(float));
  float* tempY;
  cudaMalloc(&tempY, n*sizeof(float));
  float* tempOut;
  cudaMalloc(&tempOut, n*sizeof(float));

  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, y, tempY);
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, tempX, 1, tempY, 1, tempOut));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, tempOut, out);

  cudaFree(tempX);
  cudaFree(tempY);
  cudaFree(tempOut);
}

__global__ void AsumKernelStepOne(const int n, const fp16* x, float* tempX) {
  __shared__ float xdata[ASUM_BLOCK_SIZE];
  int globalIndex = blockIdx.x * ASUM_BLOCK_SIZE + threadIdx.x;
  int localIndex = threadIdx.x;

  if(globalIndex < n) {
    xdata[localIndex] = fp16tofp32_gpu(x[globalIndex]);
    __syncthreads();

    for(int i = ASUM_BLOCK_SIZE / 2; i > 0; i>>=1) {
      if(localIndex < i) {
        xdata[localIndex] += xdata[localIndex + i];
      }
      __syncthreads();
    }

    if(localIndex == 0) {
      tempX[blockIdx.x] = xdata[localIndex];
    }

  } else {
    xdata[localIndex] = 0;
  }
}

__global__ void AsumKernelStepTwo(const int n, float* x) {
  __shared__ float xdata[ASUM_BLOCK_SIZE];
  int globalIndex = blockIdx.x * ASUM_BLOCK_SIZE + threadIdx.x;
  int localIndex = threadIdx.x;

  if(globalIndex < n) {
    xdata[localIndex] = x[globalIndex];
    __syncthreads();

    for(int i = ASUM_BLOCK_SIZE / 2; i > 0; i>>=1) {
      if(localIndex < i) {
        xdata[localIndex] += xdata[localIndex + i];
      }
      __syncthreads();
    }

    if(localIndex == 0) {
      x[blockIdx.x] = xdata[localIndex];
    }

  } else {
    xdata[localIndex] = 0;
  }
}

template <>
void caffe_gpu_asum_half<float>(const int n, const fp16* x, float* y) {
#ifdef CUSTOM_ASUM
  float* tempX;
  cudaMalloc(&tempX, ((ASUM_BLOCK_SIZE + n - 1) / ASUM_BLOCK_SIZE) * sizeof(float));
  AsumKernelStepOne<<<(ASUM_BLOCK_SIZE + n - 1) / ASUM_BLOCK_SIZE, ASUM_BLOCK_SIZE>>>(n, x, tempX);
  int ntemp = (ASUM_BLOCK_SIZE + n - 1) / ASUM_BLOCK_SIZE;
  while(ntemp > 1) {
    AsumKernelStepTwo<<<(ASUM_BLOCK_SIZE + ntemp - 1) / ASUM_BLOCK_SIZE, ASUM_BLOCK_SIZE>>>(ntemp, tempX);
    ntemp = (ASUM_BLOCK_SIZE + ntemp - 1) / ASUM_BLOCK_SIZE;
  }
  cudaMemcpy(y, tempX, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(tempX);
#else
  float* tempX;
  cudaMalloc(&tempX, n * sizeof(float));
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, tempX, 1, y));
  cudaFree(tempX);
#endif
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
  cudaMalloc(&tempX, n*sizeof(float));
  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempX);
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, tempX, 1, &tempY));
  *y = fp32tofp16(tempY);
  cudaFree(tempX);
}

__global__ void ScalKernel(const int n, const float alpha, const fp16 *x,
                            fp16* y) {
  int index = blockIdx.x * SCAL_BLOCK_SIZE + threadIdx.x;
  if(index < n) {
    y[index] = fp32tofp16_gpu(fp16tofp32_gpu(x[index]) * alpha);
  }
}

void caffe_gpu_scale_half(const int n, const float alpha, const fp16 *x,
                            fp16* y) {
  ScalKernel<<<(n + SCAL_BLOCK_SIZE - 1) / SCAL_BLOCK_SIZE, SCAL_BLOCK_SIZE>>>(n, alpha, x, y);
}

void caffe_gpu_scale_half(const int n, const double alpha, const fp16 *x,
                            fp16* y) {

  double* tempY;
  cudaMalloc(&tempY, n*sizeof(double));

  convert_to_float<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, tempY);
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, tempY, 1));
  convert_to_fp16<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, tempY, y);

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
