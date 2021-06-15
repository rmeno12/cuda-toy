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

  Matrix operator+(Matrix rhs);
  Matrix operator-(Matrix rhs);
  Matrix operator*(Matrix rhs);
  Matrix operator*(float rhs);
  float& operator()(size_t i, size_t j);
  float operator()(size_t i, size_t j) const;
  Matrix transpose();
  Matrix hadamard_product(Matrix rhs);
  Matrix collapse_horizontal_avg();

  void print();

  size_t get_rows();
  size_t get_cols();
};

#endif