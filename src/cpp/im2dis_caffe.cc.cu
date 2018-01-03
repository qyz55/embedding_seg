#include <algorithm>

#include "im2dis_caffe.h"
#include "cuda_helper.h"

namespace {

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
}

namespace caffe {

template <typename T>
__host__ __device__ T abs_t(const T num)
{
  if (num >= 0 ) return num;
  return (-num);
}

template <typename T>
__global__ void im2dis_gpu_kernel(const int n, const T* data_im, const int channels,
                                  const int height, const int width,
                                  const int kernel_h, const int kernel_w,
                                  const int pad_h, const int pad_w,
                                  const int stride_h, const int stride_w,
                                  const int dilation_h, const int dilation_w,
                                  const int height_col, const int width_col,
                                  T* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    //input_shape = {NHWC}
    //output_shape = {batch_size, output_h , output_w, kernel_size_[0] * kernel_size_[1]};
    const int h_col = index / width_col;
    const int w_col = index % width_col;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    T* data_col_ptr = data_col;
    data_col_ptr += (h_col * width_col + w_col) * kernel_h * kernel_w;
    const T* data_im_ptr = data_im;
    const int h_mid = h_offset + ((kernel_h - 1) / 2) * dilation_h;
    const int w_mid = w_offset + ((kernel_w - 1) / 2) * dilation_w;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j){
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        int mid_ptr = (h_mid * width + w_mid) * channels;
        *data_col_ptr = 0;
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width){
          int po_ptr = (h_im * width + w_im) * channels;
          for (int k = 0; k < channels; ++k){
            *data_col_ptr += abs_t(data_im_ptr[mid_ptr + k] - data_im_ptr[po_ptr + k]);
          }
        } else{
          for (int k = 0; k < channels; ++k){
            *data_col_ptr += abs_t(data_im_ptr[mid_ptr + k]);
          }
        }
        data_col_ptr ++;
      }
    }
  }
}

template <typename T>
void im2dis(const GPUDevice& d, const T* data_im, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_col) {
  // We are going to launch height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2dis_gpu_kernel<
      T><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, channels, height, width, kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, dilation_h, dilation_w, height_col, width_col,
      data_col);
  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void dis2im_gpu_kernel(const int n, const T* grad_col,
                                  const int height, const int width,
                                  const int channels, const int kernel_h,
                                  const int kernel_w, const int pad_h,
                                  const int pad_w, const int stride_h,
                                  const int stride_w, const int dilation_h,
                                  const int dilation_w, const int height_col,
                                  const int width_col, T* grad_im, const T* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    T val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int grad_col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col +
               h_col) *
                  width_col +
              w_col;
          val += grad_col[grad_col_index];
        }
      }
    }
    grad_im[index] = val;
  }
}

template <typename T>
void dis2im(const GPUDevice& d, const T* grad_col, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* grad_im, const T* data_im) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  dis2im_gpu_kernel<
      T><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, grad_col, height, width, channels, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, 
      grad_im, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
#define INSTANTIATE_IM2DIS_GPU(type)                                \
  template void im2dis<type>(                                       \
      const GPUDevice& d, const type* data_im, const int channels,  \
      const int height, const int width, const int kernel_h,        \
      const int kernel_w, const int pad_h, const int pad_w,         \
      const int stride_h, const int stride_w, const int dilation_h, \
      const int dilation_w, type* data_col);
#define INSTANTIATE_DIS2IM_GPU(type)                                \
  template void dis2im<type>(                                       \
      const GPUDevice& d, const type* grad_col, const int channels, \
      const int height, const int width, const int kernel_h,        \
      const int kernel_w, const int pad_h, const int pad_w,         \
      const int stride_h, const int stride_w, const int dilation_h, \
      const int dilation_w, type* grad_im, const type *data_im);

#define INSTANTIATE_IM2DIS_DIS2IM_GPU(type) \
  INSTANTIATE_IM2DIS_GPU(type);             \
  INSTANTIATE_DIS2IM_GPU(type);

INSTANTIATE_IM2DIS_DIS2IM_GPU(float);
INSTANTIATE_IM2DIS_DIS2IM_GPU(double);
INSTANTIATE_IM2DIS_DIS2IM_GPU(int);
INSTANTIATE_IM2DIS_DIS2IM_GPU(uint8);

#undef INSTANTIATE_IM2DIS_GPU
#undef INSTANTIATE_DIS2IM_GPU
#undef INSTANTIATE_IM2DIS_DIS2IM_GPU

}  // namespace caffe
