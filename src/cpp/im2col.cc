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

#include "im2col_caffe.h"

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

Status Im2ColShape(shape_inference::InferenceContext *c) {
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
                      output_h, output_w});
  } else if (data_format == "NHWC") {
    output_shape = c->MakeShape({batch_size, output_h, output_w,
                                 kernel_size[0] * kernel_size[1], channels});
  }
  c->set_output(0, output_shape);
  return Status::OK();
}

REGISTER_OP("Im2Col")
    .Attr("kernel_size: list(int)")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilation_rate: list(int)")
    .Attr("T: type")
    .Attr("data_format: string = 'NHWC'")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(Im2ColShape);
REGISTER_OP("Im2ColGrad")
    .Attr("kernel_size: list(int)")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilation_rate: list(int)")
    .Attr("T: type")
    .Attr("data_format: string = 'NHWC'")
    .Input("input_size: int32")
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
void GetIm2ColAttr(OpKernelConstruction *context, std::vector<T> &strides,
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
class Im2ColOp : public OpKernel {
 public:
  Im2ColOp(OpKernelConstruction *context) : OpKernel(context) {
    GetIm2ColAttr(context, strides_, kernel_size_, padding_, dilation_rate_,
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
      output_shape = {batch_size, channels, kernel_size_[0] * kernel_size_[1],
                      output_h, output_w};
    } else if (data_format_ == "NHWC") {
      throw std::runtime_error("Not Implemented");
    }
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape(output_shape), &output));

    auto Tinput = input.tensor<T, 4>();
    auto Toutput = output->tensor<T, 5>();

    auto input_data = Tinput.data();
    auto output_data = Toutput.data();

    const int input_batch_dim = channels * height * width;
    const int output_batch_dim =
        channels * output_h * output_w * kernel_size_[0] * kernel_size_[1];
    for (int b = 0; b < batch_size; ++b) {
      caffe::im2col(context->eigen_device<Device>(),
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
class Im2ColGradOp : public OpKernel {
 public:
  Im2ColGradOp(OpKernelConstruction *context) : OpKernel(context) {
    GetIm2ColAttr(context, strides_, kernel_size_, padding_, dilation_rate_,
                  data_format_);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_size = context->input(0);
    const Tensor &output_grad = context->input(1);
    OP_REQUIRES(context, output_grad.dims() == 5,
                errors::InvalidArgument("Column map must have 5 dimension.",
                                        output_grad.shape().DebugString()));
    OP_REQUIRES(context, input_size.NumElements() == 4,
                errors::InvalidArgument("input size must have 4 elements.",
                                        input_size.shape().DebugString()));

    int batch_size = output_grad.dim_size(0);
    auto Tinput_size = input_size.vec<int>();
    int ih, iw, oh, ow, channels, wsize;
    if (data_format_ == "NCHW") {
      channels = output_grad.dim_size(1);
      wsize = output_grad.dim_size(2);
      oh = output_grad.dim_size(3);
      ow = output_grad.dim_size(4);
      ih = Tinput_size(2);
      iw = Tinput_size(3);
    } else if (data_format_ == "NHWC") {
      throw std::runtime_error("Not Implemented");
    } else {
      throw std::runtime_error("Unknown data_format: " + data_format_);
    }

    OP_REQUIRES(context, kernel_size_[0] * kernel_size_[1] == wsize,
                errors::InvalidArgument("wsize does not match",
                                        output_grad.shape().DebugString()));

    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(input_size.vec<int32>(),
                                                        &input_shape));
    Tensor *input_grad = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &input_grad));

    auto Toutput_grad = output_grad.tensor<T, 5>();
    auto Tinput_grad = input_grad->tensor<T, 4>();

    auto col_data = Toutput_grad.data();
    auto im_data = Tinput_grad.data();

    const int col_batch_dim = channels * wsize * oh * ow;
    int input_grad_batch_dim = 1;
    for (int i = 1; i < input_size.NumElements(); ++i) {
      input_grad_batch_dim *= Tinput_size(i);
    }
    for (int b = 0; b < batch_size; ++b) {
      caffe::col2im(context->eigen_device<Device>(),
                    col_data + col_batch_dim * b, channels, ih, iw,
                    kernel_size_[0], kernel_size_[1], padding_[0], padding_[1],
                    strides_[0], strides_[1], dilation_rate_[0],
                    dilation_rate_[1], im_data + b * input_grad_batch_dim);
    }
  }

 private:
  std::vector<int> kernel_size_;
  std::vector<int> padding_;
  std::vector<int> strides_;
  std::vector<int> dilation_rate_;
  std::string data_format_;
};

#define REGISTER_IM2COL_DEVICE(type, device)                            \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Im2Col").Device(DEVICE_##device).TypeConstraint<type>("T"), \
      Im2ColOp<device##Device, type>);                                  \
  REGISTER_KERNEL_BUILDER(Name("Im2ColGrad")                            \
                              .Device(DEVICE_##device)                  \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("input_size"),                \
                          Im2ColGradOp<device##Device, type>);

#define REGISTER_IM2COL(type)        \
  REGISTER_IM2COL_DEVICE(type, CPU); \
  REGISTER_IM2COL_DEVICE(type, GPU);

REGISTER_IM2COL(float);
REGISTER_IM2COL(double);
REGISTER_IM2COL(int);
REGISTER_IM2COL(uint8);

#undef REGISTER_IM2COL_DEVICE
#undef REGISTER_IM2COL

}  // tensorflow
