#pragma once
#include <cuda.h>
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2 };
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
typedef struct cudaGraph_st *cudaGraph_t;
typedef struct cudaGraphExec_st *cudaGraphExec_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
enum cudaMemoryType { cudaMemoryTypeUnregistered = 0, cudaMemoryTypeHost = 1, cudaMemoryTypeDevice = 2, cudaMemoryTypeManaged = 3 };
struct cudaPointerAttributes { enum cudaMemoryType type; int device; void *devicePointer; void *hostPointer; };
struct cudaDeviceProp { char name[256]; size_t totalGlobalMem; int major; int minor; int multiProcessorCount; int integrated; int concurrentManagedAccess; size_t sharedMemPerBlock; int maxThreadsPerBlock; };
enum cudaDeviceAttr { cudaDevAttrIntegrated = 18, cudaDevAttrConcurrentManagedAccess = 89 };
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1, cudaStreamCaptureModeRelaxed = 2 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1 };
#define cudaStreamNonBlocking 1
#define cudaHostAllocDefault 0
#define cudaHostAllocMapped 2
#define cudaMemAttachGlobal 1
#define cudaEventDisableTiming 2
cudaError_t cudaMalloc(void **, size_t);
cudaError_t cudaMallocManaged(void **, size_t, unsigned int = cudaMemAttachGlobal);
cudaError_t cudaFree(void *);
cudaError_t cudaHostAlloc(void **, size_t, unsigned int);
cudaError_t cudaFreeHost(void *);
cudaError_t cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t = 0);
cudaError_t cudaMemset(void *, int, size_t);
cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t = 0);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char *cudaGetErrorString(cudaError_t);
const char *cudaGetErrorName(cudaError_t);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *, const void *);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDevice(int *);
cudaError_t cudaGetDeviceCount(int *);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *, int);
cudaError_t cudaDeviceGetAttribute(int *, enum cudaDeviceAttr, int);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaStreamCreate(cudaStream_t *);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned int);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t, enum cudaStreamCaptureMode);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t *);
cudaError_t cudaStreamIsCapturing(cudaStream_t, enum cudaStreamCaptureStatus *);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t, unsigned long long);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned int);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaMemGetInfo(size_t *, size_t *);
cudaError_t cudaMemPrefetchAsync(const void *, size_t, int, cudaStream_t = 0);
cudaError_t cudaMemAdvise(const void *, size_t, int, int);
#define cudaMemAdviseSetPreferredLocation 3
#define cudaMemAdviseSetAccessedBy 5
#define cudaCpuDeviceId (-1)
