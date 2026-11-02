#pragma once
#include <cuda_runtime.h>
typedef struct cublasContext *cublasHandle_t;
typedef int cublasStatus_t;
enum { CUBLAS_STATUS_SUCCESS = 0 };
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 } cublasOperation_t;
typedef enum { CUDA_R_16F = 2, CUDA_R_32F = 0, CUDA_R_16BF = 14 } cudaDataType;
typedef cudaDataType cudaDataType_t;
typedef enum { CUBLAS_COMPUTE_32F = 68, CUBLAS_COMPUTE_16F = 64, CUBLAS_COMPUTE_32F_FAST_16F = 74 } cublasComputeType_t;
typedef enum { CUBLAS_GEMM_DEFAULT = -1, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99 } cublasGemmAlgo_t;
typedef enum { CUBLAS_POINTER_MODE_HOST = 0 } cublasPointerMode_t;
#define CUBLAS_TF32_TENSOR_OP_MATH 4
#define CUBLAS_DEFAULT_MATH 0
cublasStatus_t cublasCreate(cublasHandle_t *);
cublasStatus_t cublasDestroy(cublasHandle_t);
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasSetMathMode(cublasHandle_t, int);
cublasStatus_t cublasSetPointerMode(cublasHandle_t, cublasPointerMode_t);
cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, cudaDataType, int, const void *, cudaDataType, int, const void *, void *, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
cublasStatus_t cublasSgemv(cublasHandle_t, cublasOperation_t, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
const char *cublasGetStatusString(cublasStatus_t);
