#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

class Matrix {
 private:
  std::vector<std::vector<float>> mat;
  size_t rows, cols;

 public:
  Matrix(size_t rows, size_t cols);
  Matrix(const Matrix& rhs);
  Matrix(std::vector<float> vals, size_t rows, size_t cols);
  Matrix& operator=(const Matrix& rhs);

  float& operator()(size_t i, size_t j);
  float operator()(size_t i, size_t j) const;

  Matrix operator+(Matrix rhs);
  Matrix operator-(Matrix rhs);
  Matrix operator*(Matrix rhs);
  Matrix operator+(float rhs);
  Matrix operator*(float rhs);
  Matrix operator-();
  Matrix transpose();
  Matrix product(Matrix rhs);
  Matrix divide(Matrix rhs);

  Matrix mean(int axis);
  static Matrix maximum(Matrix lhs, float rhs);
  static Matrix exp(Matrix input);
  static Matrix log2(Matrix input);

  void print();

  size_t get_rows();
  size_t get_cols();
};

#endif