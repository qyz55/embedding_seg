#define EIGEN_USE_THREADS

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "im2dis_caffe.h"

namespace tensorflow {

namespace {

using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

Status GetWindowOutputSizeFromDims(
    shape_inference::InferenceContext *c,
    shape_inference::DimensionHandle input_size, int kernel_size, int stride,
    int padding, int dilation_rate,
    shape_inference::DimensionHandle *output_size) {
  TF_RETURN_IF_ERROR(c->Add(input_size, 2 * padding, output_size));
  TF_RETURN_IF_ERROR(c->Subtract(
      *output_size, dilation_rate * (kernel_size - 1) + 1, output_size));
  TF_RETURN_IF_ERROR(c->Divide(*output_size, stride, false, output_size));
  TF_RETURN_IF_ERROR(c->Add(*output_size, 1, output_size));
  return Status::OK();
}

Status Im2DisShape(shape_inference::InferenceContext *c) {
  std::vector<int> strides, kernel_size, padding, dilation_rate;
  std::string data_format;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
  TF_RETURN_IF_ERROR(c->GetAttr("dilation_rate", &dilation_rate));
  TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));

  ShapeHandle conv_input_shape = c->input(0);
  DimensionHandle batch_size = c->Dim(conv_input_shape, 0);
  DimensionHandle height;
  DimensionHandle width;
  DimensionHandle channels;

  if (data_format == "NCHW") {
    channels = c->Dim(conv_input_shape, 1);
    height = c->Dim(conv_input_shape, 2);
    width = c->Dim(conv_input_shape, 3);
  } else if (data_format == "NHWC") {
    height = c->Dim(conv_input_shape, 1);
    width = c->Dim(conv_input_shape, 2);
    channels = c->Dim(conv_input_shape, 3);
  } else {
    throw std::runtime_error("Not Implemented");
  }

  DimensionHandle output_h, output_w;
  TF_RETURN_IF_ERROR(GetWindowOutputSizeFromDims(c, height, kernel_size[0],
                                                 strides[0], padding[0],
                                                 dilation_rate[0], &output_h));
  TF_RETURN_IF_ERROR(GetWindowOutputSizeFromDims(c, width, kernel_size[1],
                                                 strides[1], padding[1],
                                                 dilation_rate[1], &output_w));
  ShapeHandle output_shape;
  if (data_format == "NCHW") {
    output_shape =
        c->MakeShape({batch_size, channels, kernel_size[0] * kernel_size[1],
                      output_h , output_w});
  } else if (data_format == "NHWC") {
    output_shape = c->MakeShape({batch_size, output_h, output_w,
                                 kernel_size[0] * kernel_size[1]});
  }
  c->set_output(0, output_shape);
  return Status::OK();
}

REGISTER_OP("Im2Dis")
    .Attr("kernel_size: list(int)")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilation_rate: list(int)")
    .Attr("T: type")
    .Attr("data_format: string = 'NHWC'")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(Im2DisShape);
REGISTER_OP("Im2DisGrad")
    .Attr("kernel_size: list(int)")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilation_rate: list(int)")
    .Attr("T: type")
    .Attr("data_format: string = 'NHWC'")
    .Input("input_data: T")
    .Input("output_grad: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      ShapeHandle out;
      const Tensor *input = c->input_tensor(0);
      if (input == nullptr) {
        out = c->UnknownShapeOfRank(4);
      } else {
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      }
      c->set_output(0, out);
      return Status::OK();
    });

template <typename T>
void GetIm2DisAttr(OpKernelConstruction *context, std::vector<T> &strides,
                   std::vector<T> &kernel_size, std::vector<T> &padding,
                   std::vector<T> &dilation_rate, std::string &data_format) {
  OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
  OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
  OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
  OP_REQUIRES_OK(context, context->GetAttr("dilation_rate", &dilation_rate));
  OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));

  OP_REQUIRES(context, strides.size() == 2,
              errors::InvalidArgument(
                  "Sliding window strides field must specify 2 dimensions"));
  OP_REQUIRES(
      context, kernel_size.size() == 2,
      errors::InvalidArgument("kernel size field must specify 2 dimensions"));
  OP_REQUIRES(
      context, dilation_rate.size() == 2,
      errors::InvalidArgument("dilation rate field must specify 2 dimensions"));
  OP_REQUIRES(
      context, padding.size() == 2,
      errors::InvalidArgument("padding field must specify 2 dimensions"));
}
}

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class Im2DisOp : public OpKernel {
 public:
  Im2DisOp(OpKernelConstruction *context) : OpKernel(context) {
    GetIm2DisAttr(context, strides_, kernel_size_, padding_, dilation_rate_,
                  data_format_);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("Embedding map must have 4 dimension.",
                                        input.shape().DebugString()));

    int batch_size = input.dim_size(0);
    int height, width, channels;
    if (data_format_ == "NCHW") {
      channels = input.dim_size(1);
      height = input.dim_size(2);
      width = input.dim_size(3);
    } else if (data_format_ == "NHWC") {
      height = input.dim_size(1);
      width = input.dim_size(2);
      channels = input.dim_size(3);
    } else {
      throw std::runtime_error("Unknown data_format: " + data_format_);
    }

    const int output_h = (height + 2 * padding_[0] -
                          (dilation_rate_[0] * (kernel_size_[0] - 1) + 1)) /
                             strides_[0] +
                         1;
    const int output_w = (width + 2 * padding_[1] -
                          (dilation_rate_[1] * (kernel_size_[1] - 1) + 1)) /
                             strides_[1] +
                         1;
    TensorShape output_shape;
    if (data_format_ == "NCHW") {
      throw std::runtime_error("Not Implemented");
    } else if (data_format_ == "NHWC") {
      output_shape = {batch_size, output_h , output_w, kernel_size_[0] * kernel_size_[1]};
    }
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape(output_shape), &output));

    auto Tinput = input.tensor<T, 4>();
    auto Toutput = output->tensor<T, 4>();

    auto input_data = Tinput.data();
    auto output_data = Toutput.data();

    const int input_batch_dim = channels * height * width;
    const int output_batch_dim =
        output_h * output_w * kernel_size_[0] * kernel_size_[1];
    for (int b = 0; b < batch_size; ++b) {
      caffe::im2dis(context->eigen_device<Device>(),
                    input_data + input_batch_dim * b, channels, height, width,
                    kernel_size_[0], kernel_size_[1], padding_[0], padding_[1],
                    strides_[0], strides_[1], dilation_rate_[0],
                    dilation_rate_[1], output_data + b * output_batch_dim);
    }
  }

 private:
  std::vector<int> kernel_size_;
  std::vector<int> padding_;
  std::vector<int> strides_;
  std::vector<int> dilation_rate_;
  std::string data_format_;
};

template <typename Device, typename T>
class Im2DisGradOp : public OpKernel {
 public:
  Im2DisGradOp(OpKernelConstruction *context) : OpKernel(context) {
    GetIm2DisAttr(context, strides_, kernel_size_, padding_, dilation_rate_,
                  data_format_);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_data = context->input(0);
    const Tensor &output_grad = context->input(1);
    OP_REQUIRES(context, output_grad.dims() == 4,
                errors::InvalidArgument("Distance map must have 4 dimension.",
                                        output_grad.shape().DebugString()));
    OP_REQUIRES(context, input_data.dims() == 4,
                errors::InvalidArgument("input size must have 4 elements.",
                                        input_data.shape().DebugString()));

    int batch_size = output_grad.dim_size(0);
    int height, width, channels, wsize;
    if (data_format_ == "NCHW") {
      throw std::runtime_error("Not Implemented");
    } else if (data_format_ == "NHWC") {
      height = input_data.dim_size(1);
      width = input_data.dim_size(2);
      channels = input_data.dim_size(3);
      wsize = output_grad.dim_size(3);
    } else {
      throw std::runtime_error("Unknown data_format: " + data_format_);
    }

    OP_REQUIRES(context, kernel_size_[0] * kernel_size_[1] == wsize,
                errors::InvalidArgument("wsize does not match",
                                        output_grad.shape().DebugString()));
    std::vector<int> input_size;
    for (int i = 0; i < input_data.dims(); ++i){
      input_size.push_back(input_data.dim_size(i));
    }
    TensorShape output_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(input_size,
                                                        &output_shape));
    Tensor *output = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto Tinput = input_data.tensor<T, 4>();
    auto Toutput_grad = output_grad.tensor<T, 4>();
    auto Tinput_grad = output->tensor<T, 4>();

    auto col_grad = Toutput_grad.data();
    auto im_grad = Tinput_grad.data();
    auto im_data = Tinput.data();

    const int input_batch_dim = channels * height * width;
    int output_batch_dim = 1;
    for (int i = 1; i < output_grad.dims(); ++i) {
      output_batch_dim *= output_grad.dim_size(i);
    }
    for (int b = 0; b < batch_size; ++b) {
      caffe::dis2im(context->eigen_device<Device>(),
                    col_grad + output_batch_dim * b, channels, height, width,
                    kernel_size_[0], kernel_size_[1], padding_[0], padding_[1],
                    strides_[0], strides_[1], dilation_rate_[0],
                    dilation_rate_[1], im_grad + b * input_batch_dim, im_data + b * input_batch_dim);
    }
  }

 private:
  std::vector<int> kernel_size_;
  std::vector<int> padding_;
  std::vector<int> strides_;
  std::vector<int> dilation_rate_;
  std::string data_format_;
};

#define REGISTER_IM2DIS_DEVICE(type, device)                            \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Im2Dis").Device(DEVICE_##device).TypeConstraint<type>("T"), \
      Im2DisOp<device##Device, type>);                                  \
  REGISTER_KERNEL_BUILDER(Name("Im2DisGrad")                            \
                              .Device(DEVICE_##device)                  \
                              .TypeConstraint<type>("T")                \
                              ,Im2DisGradOp<device##Device, type>);
                                              
                          

#define REGISTER_IM2DIS(type)        \
  REGISTER_IM2DIS_DEVICE(type, CPU); \
  REGISTER_IM2DIS_DEVICE(type, GPU);

REGISTER_IM2DIS(double);
REGISTER_IM2DIS(float);
REGISTER_IM2DIS(int);
REGISTER_IM2DIS(uint8);

#undef REGISTER_IM2DIS_DEVICE
#undef REGISTER_IM2DIS

}  // tensorflow
