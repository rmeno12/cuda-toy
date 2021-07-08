#include "matrix.hpp"

__global__ void matmul_k(float* lhs, float* rhs, float* res, size_t rows,
                         size_t mid, size_t cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    float sum = 0;
    for (int i = 0; i < mid; i++) {
      sum += lhs[row * mid + i] * rhs[i * cols + col];
    }
    res[row * cols + col] = sum;
  }
}

__global__ void matadd_k(float* lhs, float* rhs, size_t rows, size_t cols,
                         bool sub, bool broadcast) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    lhs[row * cols + col] +=
        (sub ? -1 : 1) * (broadcast ? rhs[row] : rhs[row * cols + col]);
  }
}

__global__ void matprod_k(float* lhs, float* rhs, size_t rows, size_t cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    lhs[row * cols + col] *= rhs[row * cols + col];
  }
}

__global__ void matdiv_k(float* lhs, float* rhs, size_t rows, size_t cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    lhs[row * cols + col] /= rhs[row * cols + col];
  }
}

__global__ void mataddscal_k(float* lhs, float rhs, size_t rows, size_t cols,
                             bool sub) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    lhs[row * cols + col] += (sub ? -1 : 1) * rhs;
  }
}

__global__ void matprodscal_k(float* lhs, float rhs, size_t rows, size_t cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    lhs[row * cols + col] *= rhs;
  }
}

__global__ void matdivscal_k(float* lhs, float rhs, size_t rows, size_t cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    lhs[row * cols + col] /= rhs;
  }
}