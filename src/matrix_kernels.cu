#include "matrix.hpp"
#include "stdio.h"

__global__ void augment_k(float* lhs, float* rhs, float* res, size_t lrows,
                          size_t lcols, size_t rrows, size_t rcols, int axis) {
  size_t rows, cols;
  if (axis == 0) {
    rows = lrows + rrows;
    cols = lcols;
  } else if (axis == 1) {
    rows = lrows;
    cols = lcols + rcols;
  }

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    if (axis == 0) {
      if (row < lrows) {
        res[row * cols + col] = lhs[row * lcols + col];
      } else {
        res[row * cols + col] = rhs[(row - lrows) * rcols + col];
      }
    } else if (axis == 1) {
      if (col < lcols) {
        res[row * cols + col] = lhs[row * lcols + col];
      } else {
        res[row * cols + col] = rhs[row * rcols + (col - lcols)];
      }
    }
  }
}

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

__global__ void mattranspose_k(float* mat, float* res, size_t rows,
                               size_t cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    res[row * cols + col] = mat[col * rows + row];
  }
}