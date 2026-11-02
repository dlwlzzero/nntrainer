#pragma once
#include <cstddef>
typedef struct _nvrtcProgram *nvrtcProgram;
typedef int nvrtcResult;
enum { NVRTC_SUCCESS = 0 };
nvrtcResult nvrtcCreateProgram(nvrtcProgram *, const char *, const char *, int, const char *const *, const char *const *);
nvrtcResult nvrtcCompileProgram(nvrtcProgram, int, const char *const *);
nvrtcResult nvrtcGetPTXSize(nvrtcProgram, size_t *);
nvrtcResult nvrtcGetPTX(nvrtcProgram, char *);
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram, size_t *);
nvrtcResult nvrtcGetCUBIN(nvrtcProgram, char *);
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram, size_t *);
nvrtcResult nvrtcGetProgramLog(nvrtcProgram, char *);
nvrtcResult nvrtcDestroyProgram(nvrtcProgram *);
nvrtcResult nvrtcAddNameExpression(nvrtcProgram, const char *);
nvrtcResult nvrtcGetLoweredName(nvrtcProgram, const char *, const char **);
const char *nvrtcGetErrorString(nvrtcResult);
