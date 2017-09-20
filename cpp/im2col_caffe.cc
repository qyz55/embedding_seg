#include <algorithm>
#include <vector>

#include "im2col_caffe.h"

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
void im2col(const CPUDevice& d, const T* data_im, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_col) {
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename T>
void col2im(const CPUDevice& d, const T* data_col, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_im) {
  std::fill_n(data_im, height * width * channels, 0.0);
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
#define INSTANTIATE_IM2COL_CPU(type)                                \
  template void im2col<type>(                                       \
      const CPUDevice& d, const type* data_im, const int channels,  \
      const int height, const int width, const int kernel_h,        \
      const int kernel_w, const int pad_h, const int pad_w,         \
      const int stride_h, const int stride_w, const int dilation_h, \
      const int dilation_w, type* data_col);
#define INSTANTIATE_COL2IM_CPU(type)                                \
  template void col2im<type>(                                       \
      const CPUDevice& d, const type* data_col, const int channels, \
      const int height, const int width, const int kernel_h,        \
      const int kernel_w, const int pad_h, const int pad_w,         \
      const int stride_h, const int stride_w, const int dilation_h, \
      const int dilation_w, type* data_im);

#define INSTANTIATE_IM2COL_COL2IM_CPU(type) \
  INSTANTIATE_IM2COL_CPU(type);             \
  INSTANTIATE_COL2IM_CPU(type);

INSTANTIATE_IM2COL_COL2IM_CPU(float);
INSTANTIATE_IM2COL_COL2IM_CPU(double);
INSTANTIATE_IM2COL_COL2IM_CPU(int);

#undef INSTANTIATE_IM2COL_CPU
#undef INSTANTIATE_COL2IM_CPU
#undef INSTANTIATE_IM2COL_COL2IM_CPU

}  // namespace caffe
