#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace caffe {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
void im2col(const CPUDevice& d, const T* data_im, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_col);

template <typename T>
void im2col(const GPUDevice& d, const T* data_im, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_col);

template <typename T>
void col2im(const CPUDevice& d, const T* data_col, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_im);

template <typename T>
void col2im(const GPUDevice& d, const T* data_col, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
