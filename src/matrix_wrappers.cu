#include "matrix.hpp"
#include "matrix_kernels.cuh"

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

void matadd_wrapper(float* lhs, float* rhs, size_t rows, size_t cols, bool sub,
                    bool broadcast) {
  float *d_l, *d_r;
  cudaMalloc(&d_l, sizeof(float) * rows * cols);
  cudaMalloc(&d_r, sizeof(float) * rows * (broadcast ? 1 : cols));

  cudaMemcpy(d_l, lhs, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, rhs, sizeof(float) * rows * (broadcast ? 1 : cols),
             cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  matadd_k<<<gridsize, blocksize>>>(d_l, d_r, rows, cols, sub, broadcast);
  cudaDeviceSynchronize();

  cudaMemcpy(lhs, d_l, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
  cudaFree(d_r);
}

void matprod_wrapper(float* lhs, float* rhs, size_t rows, size_t cols) {
  float *d_l, *d_r;
  cudaMalloc(&d_l, sizeof(float) * rows * cols);
  cudaMalloc(&d_r, sizeof(float) * rows * cols);

  cudaMemcpy(d_l, lhs, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, rhs, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  matprod_k<<<gridsize, blocksize>>>(d_l, d_r, rows, cols);
  cudaDeviceSynchronize();

  cudaMemcpy(lhs, d_l, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
  cudaFree(d_r);
}

void matdiv_wrapper(float* lhs, float* rhs, size_t rows, size_t cols) {
  float *d_l, *d_r;
  cudaMalloc(&d_l, sizeof(float) * rows * cols);
  cudaMalloc(&d_r, sizeof(float) * rows * cols);

  cudaMemcpy(d_l, lhs, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, rhs, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  matdiv_k<<<gridsize, blocksize>>>(d_l, d_r, rows, cols);
  cudaDeviceSynchronize();

  cudaMemcpy(lhs, d_l, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
  cudaFree(d_r);
}

void mataddscal_wrapper(float* lhs, float rhs, size_t rows, size_t cols,
                        bool sub) {
  float* d_l;
  cudaMalloc(&d_l, sizeof(float) * rows * cols);

  cudaMemcpy(d_l, lhs, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  mataddscal_k<<<gridsize, blocksize>>>(d_l, rhs, rows, cols, sub);
  cudaDeviceSynchronize();

  cudaMemcpy(lhs, d_l, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
}

void matprodscal_wrapper(float* lhs, float rhs, size_t rows, size_t cols) {
  float* d_l;
  cudaMalloc(&d_l, sizeof(float) * rows * cols);

  cudaMemcpy(d_l, lhs, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  matprodscal_k<<<gridsize, blocksize>>>(d_l, rhs, rows, cols);
  cudaDeviceSynchronize();

  cudaMemcpy(lhs, d_l, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
}

void matdivscal_wrapper(float* lhs, float rhs, size_t rows, size_t cols) {
  float* d_l;
  cudaMalloc(&d_l, sizeof(float) * rows * cols);

  cudaMemcpy(d_l, lhs, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  matdivscal_k<<<gridsize, blocksize>>>(d_l, rhs, rows, cols);
  cudaDeviceSynchronize();

  cudaMemcpy(lhs, d_l, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
}