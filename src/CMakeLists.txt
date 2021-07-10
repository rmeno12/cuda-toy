file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${cuda-toy_SOURCE_DIR}/include/*")

add_library(cuda-toy-library 
  matrix.cpp 
  matrix_kernels.cu
  matrix_wrappers.cu
  mnist.cpp
  mnist_model.cpp
  linear_relu_layer.cpp
  linear_sigmoid_layer.cpp
  linear_softmax_layer.cpp
  ${HEADER_LIST})

target_include_directories(cuda-toy-library PUBLIC ../include)

add_executable(
  gpu-cuda-toy
  main.cpp
)

target_compile_options(gpu-cuda-toy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -gencode arch=compute_60,code=sm_60 -std=c++17>)

target_link_libraries(
  gpu-cuda-toy PRIVATE cuda-toy-library
)