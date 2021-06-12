#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

class Matrix {
 private:
  std::vector<std::vector<float>> mat;
  size_t rows, cols;

 public:
  Matrix(size_t, size_t);
  Matrix(const Matrix&);
  Matrix& operator=(const Matrix&);
  ~Matrix();

  Matrix operator+(Matrix&);
  Matrix operator*(Matrix&);
  float& operator()(size_t, size_t);
  float operator()(size_t, size_t) const;
};

#endif