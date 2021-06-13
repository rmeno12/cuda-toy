#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

class Matrix {
 private:
  std::vector<std::vector<float>> mat;
  size_t rows, cols;

 public:
  Matrix(size_t, size_t);
  Matrix(const Matrix&);
  Matrix& operator=(const Matrix&);

  Matrix operator+(Matrix);
  Matrix operator-(Matrix);
  Matrix operator*(Matrix);
  Matrix operator*(float);
  float& operator()(size_t, size_t);
  float operator()(size_t, size_t) const;
  Matrix transpose();
  Matrix hadamard_product(Matrix);

  size_t get_rows();
  size_t get_cols();
};

#endif