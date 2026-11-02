// stub for a -fsyntax-only check of nntrainer/cuda sources without a CUDA toolkit
#pragma once
#include <cstddef>
#include <cstdint>
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef unsigned long long CUdeviceptr;
typedef int CUresult;
typedef int CUjit_option;
enum { CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1 };
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
int cuInit(unsigned int);
int cuDeviceGet(CUdevice *, int);
int cuDeviceGetCount(int *);
int cuDeviceGetAttribute(int *, int, CUdevice);
int cuDeviceGetName(char *, int, CUdevice);
int cuDevicePrimaryCtxRetain(CUcontext *, CUdevice);
int cuDevicePrimaryCtxRelease(CUdevice);
int cuCtxSetCurrent(CUcontext);
int cuCtxGetCurrent(CUcontext *);
int cuCtxSynchronize();
int cuModuleLoadDataEx(CUmodule *, const void *, unsigned int, CUjit_option *, void **);
int cuModuleLoadData(CUmodule *, const void *);
int cuModuleGetFunction(CUfunction *, CUmodule, const char *);
int cuModuleUnload(CUmodule);
int cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **);
int cuGetErrorString(CUresult, const char **);
int cuGetErrorName(CUresult, const char **);
int cuMemAlloc(CUdeviceptr *, size_t);
int cuMemFree(CUdeviceptr);
int cuFuncGetAttribute(int *, int, CUfunction);
int cuFuncSetAttribute(CUfunction, int, int);
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8
#define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES 1
