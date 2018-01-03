CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(embedding)

set(GOOGLE_CUDA TRUE)
add_definitions(-DGOOGLE_CUDA)

find_package(CUDA QUIET REQUIRED)

# compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${OpenMP_CXX_FLAGS} -Wall -fPIC")
SET(CUDA_NVCC_FLAGS " -std=c++11")
message(STATUS ${CMAKE_CXX_FLAGS})
message(STATUS ${CUDA_INCLUDE_DIRS})
message(STATUS ${CUDA_NVCC_FLAGS})


# TensorFlow dependencies
SET(TF_INC "/home/jianzhun/anaconda3/envs/tf/lib/python3.6/site-packages/tensorflow/include/")

# target
include_directories(${TF_INC})
cuda_add_library(embedding SHARED
    #im2dis.cc
    #im2dis_caffe.cc
    #im2dis_caffe.cc.cu
    #im2dis_caffe.h
    im2col.cc
    im2col_caffe.cc
    im2col_caffe.cc.cu
    im2col_caffe.h
)
