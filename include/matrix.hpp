#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

#define BLOCK_SIZE 16

class Matrix {
 private:
  float** mat;
  size_t rows, cols;

 public:
  Matrix(size_t rows, size_t cols);
  Matrix(const Matrix& rhs);
  Matrix(std::vector<float> vals, size_t rows, size_t cols);
  Matrix(float** vals, size_t rows, size_t cols);
  ~Matrix();
  Matrix& operator=(const Matrix& rhs);

  void augment(const Matrix& other, int axis);

  float& operator()(size_t i, size_t j);
  float operator()(size_t i, size_t j) const;

  const Matrix operator-() const;

  Matrix& operator+=(const Matrix& rhs);
  Matrix& operator-=(const Matrix& rhs);

  const Matrix operator+(const Matrix& rhs) const;
  const Matrix operator-(const Matrix& rhs) const;
  const Matrix operator*(const Matrix& rhs) const;
  const Matrix product(const Matrix& rhs) const;
  const Matrix divide(const Matrix& rhs) const;

  Matrix& operator+=(float rhs);
  Matrix& operator-=(float rhs);
  Matrix& operator*=(float rhs);
  Matrix& operator/=(float rhs);

  const Matrix operator+(float rhs) const;
  const Matrix operator-(float rhs) const;
  const Matrix operator*(float rhs) const;
  const Matrix operator/(float rhs) const;

  const Matrix transpose() const;
  const Matrix mean(int axis) const;
  const Matrix sum(int axis) const;
  const Matrix max(int axis) const;
  const Matrix min(int axis) const;

  static const Matrix maximum(const Matrix& lhs, float rhs);
  static const Matrix exp(const Matrix& input);
  static const Matrix log2(const Matrix& input);

  void print() const;

  size_t get_rows() const;
  size_t get_cols() const;
};

const Matrix operator+(float lhs, const Matrix& rhs);
const Matrix operator-(float lhs, const Matrix& rhs);
const Matrix operator*(float lhs, const Matrix& rhs);
const Matrix operator/(float lhs, const Matrix& rhs);

void augment_wrapper(float* lhs, float* rhs, float* res, size_t lrows,
                     size_t lcols, size_t rrows, size_t rcols, int axis);

void matmul_wrapper(float* lhs, float* rhs, float* res, size_t rows, size_t mid,
                    size_t cols);
void matadd_wrapper(float* lhs, float* rhs, size_t rows, size_t cols, bool sub,
                    bool broadcast);
void matprod_wrapper(float* lhs, float* rhs, size_t rows, size_t cols);
void matdiv_wrapper(float* lhs, float* rhs, size_t rows, size_t cols);

void mataddscal_wrapper(float* lhs, float rhs, size_t rows, size_t cols,
                        bool sub);
void matprodscal_wrapper(float* lhs, float rhs, size_t rows, size_t cols);
void matdivscal_wrapper(float* lhs, float rhs, size_t rows, size_t cols);

void mattranspose_wrapper(float* mat, float* res, size_t rows, size_t cols);

#endif