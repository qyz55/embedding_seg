#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "tensorflow/core/platform/default/logging.h"

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                                \
  /* Code block avoids redefinition of cudaError_t error */                  \
  do {                                                                       \
    cudaError_t error = condition;                                           \
    if (error != cudaSuccess) LOG(ERROR) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
