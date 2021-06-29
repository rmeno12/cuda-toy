file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${cuda-toy_SOURCE_DIR}/include/*.hpp")

add_library(cuda-toy-library 
  matrix.cpp 
  linear_relu_layer.cpp
  linear_sigmoid_layer.cpp
  linear_softmax_layer.cpp
  ${HEADER_LIST})

target_include_directories(cuda-toy-library PUBLIC ../include)

add_executable(
  cuda-toy
  main.cpp
)

target_link_libraries(cuda-toy PRIVATE cuda-toy-library)