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

void matmul_wrapper(float* lhs, float* rhs, float* res, size_t rows, size_t mid,
                    size_t cols) {
  float *d_l, *d_r, *d_res;
  cudaMalloc(&d_l, sizeof(float) * rows * mid);
  cudaMalloc(&d_r, sizeof(float) * mid * cols);
  cudaMalloc(&d_res, sizeof(float) * rows * cols);

  cudaMemcpy(d_l, lhs, sizeof(float) * rows * mid, cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, rhs, sizeof(float) * mid * cols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  matmul_k<<<gridsize, blocksize>>>(d_l, d_r, d_res, rows, mid, cols);
  cudaDeviceSynchronize();

  cudaMemcpy(res, d_res, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
  cudaFree(d_r);
  cudaFree(d_res);
}