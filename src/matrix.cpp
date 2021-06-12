#include "matrix.h"

#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
  mat.resize(rows);
  for (auto i : mat) {
    i.resize(cols);
  }
}

Matrix::Matrix(const Matrix& rhs) {
  mat = rhs.mat;
  rows = rhs.rows;
  cols = rhs.cols;
}

Matrix::~Matrix() {}

Matrix Matrix::operator+(Matrix& rhs) {
  if (rhs.rows != rows || rhs.cols != cols) {
    throw std::invalid_argument(
        "Matrices must be the same size to be added together.");
  }

  Matrix out(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out(i, j) = mat[i][j] + rhs(i, j);
    }
  }

  return out;
}

Matrix Matrix::operator*(Matrix& rhs) {
  // if (rhs.rows != rows || rhs.cols != cols) {
  //   throw std::invalid_argument(
  //       "Matrices must be the same size to be added together.");
  // }

  Matrix out(rows, rhs.cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < rhs.cols; j++) {
      out(i, j) = 0;
      for (size_t k = 0; k < cols; k++) {
        out(i, j) += (*this)(i, k) * rhs(k, j);
      }
    }
  }

  return out;
}

float& Matrix::operator()(size_t i, size_t j) {
  if (i >= rows || j >= cols) {
    throw;
  }

  return mat[i][j];
}

float Matrix::operator()(size_t i, size_t j) const {
  if (i >= rows || j >= cols) {
    throw;
  }

  return mat[i][j];
}