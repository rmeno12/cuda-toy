#include "matrix.hpp"

#define BLOCK_SIZE 16

__global__ void matmul_k(float* lhs, float* rhs, float* result, int rows,
                         int mid, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    int sum = 0;
    for (int i = 0; i < mid; i++) {
      sum += lhs[row * mid + i] * rhs[i * cols + col];
    }
    result[row * cols + col] = sum;
  }
}

Matrix matmul(const Matrix& lhs, const Matrix& rhs) {
  int rows = lhs.get_rows();
  int lcols = lhs.get_cols();
  int rrows = rhs.get_rows();
  int cols = rhs.get_cols();
  float* larr = new float[rows * lcols];
  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < lcols; j++) {
      larr[i * rows + j] = lhs(i, j);
    }
  }
  float* rarr = new float[rrows * cols];
  for (auto i = 0; i < rrows; i++) {
    for (auto j = 0; j < cols; j++) {
      rarr[i * rrows + j] = rhs(i, j);
    }
  }

  float *d_l, *d_r, *d_res;
  cudaMalloc(&d_l, sizeof(float) * rows * lcols);
  cudaMalloc(&d_r, sizeof(float) * rrows * cols);
  cudaMalloc(&d_res, sizeof(float) * rows * cols);

  cudaMemcpy(d_l, larr, sizeof(float) * rows * lcols, cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, rarr, sizeof(float) * rrows * cols, cudaMemcpyHostToDevice);

  uint grid_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint grid_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridsize(grid_x, grid_y);
  dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);

  matmul_k<<<gridsize, blocksize>>>(d_l, d_r, d_res, rows, lcols, cols);
  cudaDeviceSynchronize();

  cudaFree(d_l);
  cudaFree(d_r);
  float* res = new float[rows * cols];
  cudaMemcpy(res, d_res, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(d_res);
  delete[] larr;
  delete[] rarr;

  std::vector<float> vals;
  for (auto i = 0; i < rows * cols; i++) {
    vals.push_back(res[i]);
  }
  delete[] res;

  return Matrix(vals, rows, cols);
}

int main() {
  // thing
  Matrix A({1, 2, 3, 4}, 2, 2);
  Matrix B({2, 3, 3, 2}, 2, 2);
  Matrix C = matmul(A, B);

  A.print();
  B.print();
  C.print();
}