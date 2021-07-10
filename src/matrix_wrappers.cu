#include "matrix.hpp"
#include "matrix_kernels.cuh"

void augment_wrapper(float* lhs, float* rhs, float* res, size_t lrows,
                     size_t lcols, size_t rrows, size_t rcols, int axis) {
  size_t rows, cols;
  if (axis == 0) {
    rows = lrows + rrows;
    cols = lcols;
  } else if (axis == 1) {
    rows = lrows;
    cols = lcols + rcols;
  }

  float *d_l, *d_r, *d_res;
  cudaMalloc(&d_l, sizeof(float) * lrows * lcols);
  cudaMalloc(&d_r, sizeof(float) * rrows * rcols);
  cudaMalloc(&d_res, sizeof(float) * rows * cols);

  cudaMemcpy(d_l, lhs, sizeof(float) * lrows * lcols, cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, rhs, sizeof(float) * rrows * rcols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  augment_k<<<gridsize, blocksize>>>(d_l, d_r, d_res, lrows, lcols, rrows,
                                     rcols, axis);

  cudaMemcpy(res, d_res, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
  cudaFree(d_r);
  cudaFree(d_res);
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

  cudaMemcpy(lhs, d_l, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_l);
}

void mattranspose_wrapper(float* mat, float* res, size_t rows, size_t cols) {
  float *d_mat, *d_res;
  cudaMalloc(&d_mat, sizeof(float) * rows * cols);
  cudaMalloc(&d_res, sizeof(float) * rows * cols);

  cudaMemcpy(d_mat, mat, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, res, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  mattranspose_k<<<gridsize, blocksize>>>(d_mat, d_res, rows, cols);

  cudaMemcpy(res, d_res, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_mat);
  cudaFree(d_res);
}