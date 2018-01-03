#ifndef _CAFFE_UTIL_IM2DIS_HPP_
#define _CAFFE_UTIL_IM2DIS_HPP_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"

namespace caffe {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

using tensorflow::uint8;

template <typename T>
void im2dis(const CPUDevice& d, const T* data_im, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_col);

template <typename T>
void im2dis(const GPUDevice& d, const T* data_im, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_col);

template <typename T>
void dis2im(const CPUDevice& d, const T* grad_col, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* grad_im,const T* data_im);

template <typename T>
void dis2im(const GPUDevice& d, const T* grad_col, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* grad_im, const T* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
