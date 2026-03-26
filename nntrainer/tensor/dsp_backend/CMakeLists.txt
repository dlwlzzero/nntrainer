cmake_minimum_required(VERSION 3.14.3)
project(hmx_dsp C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Include Hexagon SDK CMake functions
if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    message(FATAL_ERROR "HEXAGON_SDK_ROOT not set. Please set it to point to Hexagon SDK installation.")
endif()

set(OS_TYPE "DSP")

# Common includes
set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/dsp
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)
include_directories(${common_incs})

# DSP static library
add_library(hmx_dsp STATIC
    dsp/commu.c
    dsp/power.c
    dsp/hmx_mgr.c
    dsp/worker_pool.c
    dsp/mmap_mgr.cc
    dsp/op_executor.cc
    dsp/op_tests.cc
    dsp/vtcm_mgr.cc
    dsp/ops/flash_attn_sp_hdim.c
    dsp/ops/flash_attn.c
    dsp/ops/mat_mul.c
    dsp/ops/mm_benchmark.c
    dsp/ops/precompute_table.c
    dsp/ops/rms_norm.c
    dsp/ops/qk_attention_kqt.c
)
build_idl(include/htp_ops.idl hmx_dsp)

# Enable HMX inline assembly
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mhmx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mhmx")

# Set target architecture
set_target_properties(hmx_dsp PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

message("Building HMX DSP library for ${HEXAGON_ARCH}")
message("Output: ${CMAKE_CURRENT_BINARY_DIR}/libhmx_dsp.a")