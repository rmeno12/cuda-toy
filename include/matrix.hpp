#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

class Matrix {
 private:
  float** mat;
  size_t rows, cols;

 public:
  Matrix(size_t rows, size_t cols);
  Matrix(const Matrix& rhs);
  Matrix(std::vector<float> vals, size_t rows, size_t cols);
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

  Matrix& operator+=(const float& rhs);
  Matrix& operator-=(const float& rhs);
  Matrix& operator*=(const float& rhs);
  Matrix& operator/=(const float& rhs);

  const Matrix operator+(const float& rhs) const;
  const Matrix operator-(const float& rhs) const;
  const Matrix operator*(const float& rhs) const;
  const Matrix operator/(const float& rhs) const;

  const Matrix transpose() const;
  const Matrix mean(const int& axis) const;
  const Matrix sum(const int& axis) const;
  static const Matrix maximum(const Matrix& lhs, const float& rhs);
  static const Matrix exp(const Matrix& input);
  static const Matrix log2(const Matrix& input);
  static float sum(const Matrix& input);

  void print() const;

  size_t get_rows() const;
  size_t get_cols() const;
};

const Matrix operator+(const float& lhs, const Matrix& rhs);
const Matrix operator-(const float& lhs, const Matrix& rhs);
const Matrix operator*(const float& lhs, const Matrix& rhs);
const Matrix operator/(const float& lhs, const Matrix& rhs);

#endif