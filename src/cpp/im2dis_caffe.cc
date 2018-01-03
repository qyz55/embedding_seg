#include <algorithm>
#include <vector>
#include <iostream>
#include "im2dis_caffe.h"

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
//input_shape = {NHWC}
//output_shape = {batch_size, output_h, output_w, kernel_size_[0] * kernel_size_[1]};
template <typename T>
T abs_t(const T num)
{
  if (num >= 0 ) return num;
  return (-num);
}

template <typename T>
void im2dis(const CPUDevice& d, const T* data_im, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* data_col) {
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int input_row0 = -pad_h;
  int input_col0 = -pad_w;
  for (int output_rows = output_h; output_rows; output_rows--) {
    for (int output_col = output_w; output_col; output_col--) {
      int input_row = input_row0;
      int input_col = input_col0;
      int center_row = input_row0 + ((kernel_h - 1) / 2) * dilation_h ;
      int center_col = input_col0 + ((kernel_w - 1) / 2) * dilation_w ;
      int ptc = center_row * width * channels + center_col * channels;
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++, input_row += dilation_h) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++, input_col += dilation_w) {
            int pti = input_row * width * channels + input_col * channels;
            if (is_a_ge_zero_and_a_lt_b(input_col, width) && is_a_ge_zero_and_a_lt_b(input_row, height)){
              for (int channel = 0; channel < channels; ++channel){	
                *data_col += abs_t(data_im[ptc + channel] - data_im[pti + channel]);
              }
              //printf("A %f %d %d %d %d %d %d\n", data_col[0] , input_row, input_col, center_row, center_col, ptc, pti);
            } else{
              for (int channel = 0; channel < channels; ++channel){
                *data_col += abs_t(data_im[ptc + channel]);
              }
              //printf("B %f %d %d %d %d %d %d\n", data_col[0] , input_row, input_col, center_row, center_col, ptc, pti);
            }
            
            data_col++;
        }
        input_col = input_col0;
      }
      input_col0 += stride_w;
    }
    input_row0 +=stride_h;
    input_col0 = -pad_w;
  }
}

template <typename T>
void dis2im(const CPUDevice& d, const T* grad_col, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, const int dilation_h,
            const int dilation_w, T* grad_im,const T* data_im) {
  //data_im=[h][w][channel]
  //grad_im=[h][w][channel]
  //grad_col=[output_h][output_w][kernel[0]* kernel[1]]
  std::fill_n(grad_im, height * width * channels, 0.0);
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int input_row0 = -pad_h;
  int input_col0 = -pad_w;
  for (int output_rows = output_h; output_rows; output_rows--) {
    for (int output_col = output_w; output_col; output_col--) {
      int input_row = input_row0;
      int input_col = input_col0;
      int center_row = input_row0 + ((kernel_h - 1) / 2) * dilation_h ;
      int center_col = input_col0 + ((kernel_w - 1) / 2) * dilation_w ;
      int ptc = center_row * width * channels + center_col * channels;
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++, input_row += dilation_h) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++, input_col += dilation_w) {
            int pti = input_row * width * channels + input_col * channels;
            if (is_a_ge_zero_and_a_lt_b(input_col, width) && is_a_ge_zero_and_a_lt_b(input_row, height)){
              for (int channel = 0; channel < channels; ++channel){
                if (data_im[ptc + channel] > data_im[pti + channel]){
                  grad_im[pti + channel] -= *grad_col;
                  grad_im[ptc + channel] += *grad_col;
                } else if (data_im[ptc + channel] < data_im[pti + channel]){
                  grad_im[pti + channel] += *grad_col;
                  grad_im[ptc + channel] -= *grad_col;
                }
                //if (ptc == 0 || pti == 0) {printf("%d %d %d %d ", ptc, pti, input_row, input_col);
                //std::cout<<data_im[ptc]<<" "<<data_im[pti]<<std::endl;
             	//std::cout<<grad_im[ptc]<<" "<<grad_im[pti]<<std::endl;}
              }
            } else{
              for (int channel = 0; channel < channels; ++channel){
                if (data_im[ptc + channel] > 0){
                  grad_im[ptc + channel] += *grad_col;
                } else if (data_im[ptc + channel] < 0){
                  grad_im[ptc + channel] -= *grad_col;
                } 
              }
              //if (ptc == 0) {printf("%d %d %d ", ptc,input_row, input_col); 
              //std::cout<<data_im[ptc]<<std::endl;
          	  //std::cout<<grad_im[ptc]<<std::endl;}
            }
            grad_col++;
        }
        input_col = input_col0;
      }
      input_col0 += stride_w;
    }
    input_row0 +=stride_h;
    input_col0 = -pad_w;
  }
}

// Explicit instantiation
#define INSTANTIATE_IM2DIS_CPU(type)                                \
  template void im2dis<type>(                                       \
      const CPUDevice& d, const type* data_im, const int channels,  \
      const int height, const int width, const int kernel_h,        \
      const int kernel_w, const int pad_h, const int pad_w,         \
      const int stride_h, const int stride_w, const int dilation_h, \
      const int dilation_w, type* data_col);
#define INSTANTIATE_DIS2IM_CPU(type)                                \
  template void dis2im<type>(                                       \
      const CPUDevice& d, const type* data_col, const int channels, \
      const int height, const int width, const int kernel_h,        \
      const int kernel_w, const int pad_h, const int pad_w,         \
      const int stride_h, const int stride_w, const int dilation_h, \
      const int dilation_w, type* grad_im,const type* data_im);

#define INSTANTIATE_IM2DIS_DIS2IM_CPU(type) \
  INSTANTIATE_IM2DIS_CPU(type);             \
  INSTANTIATE_DIS2IM_CPU(type);

INSTANTIATE_IM2DIS_DIS2IM_CPU(float);
INSTANTIATE_IM2DIS_DIS2IM_CPU(double);
INSTANTIATE_IM2DIS_DIS2IM_CPU(int);
INSTANTIATE_IM2DIS_DIS2IM_CPU(uint8);

#undef INSTANTIATE_IM2DIS_CPU
#undef INSTANTIATE_DIS2IM_CPU
#undef INSTANTIATE_IM2DIS_DIS2IM_CPU

}  // namespace caffe
