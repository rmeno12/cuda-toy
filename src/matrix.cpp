#include "matrix.hpp"

#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
  mat.resize(rows);
  for (auto i = 0; i < rows; i++) {
    mat[i] = {};
    mat[i].resize(cols);
  }
}

Matrix::Matrix(const Matrix& rhs) {
  mat = rhs.mat;
  rows = rhs.rows;
  cols = rhs.cols;
}

Matrix& Matrix::operator=(const Matrix& rhs) {
  if (this == &rhs) return *this;

  if (rows != rhs.rows) {
    mat.resize(rhs.rows);
  }

  if (cols != rhs.cols) {
    for (auto i : mat) {
      i.resize(rhs.cols);
    }
  }

  for (size_t i = 0; i < rhs.rows; i++) {
    for (size_t j = 0; j < rhs.cols; j++) {
      mat[i][j] = rhs(i, j);
    }
  }

  return *this;
}

Matrix Matrix::operator+(Matrix rhs) {
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

Matrix Matrix::operator-(Matrix rhs) {
  if (rhs.rows != rows || rhs.cols != cols) {
    throw std::invalid_argument(
        "Matrices must be the same size to be subtracted.");
  }

  Matrix out(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out(i, j) = mat[i][j] - rhs(i, j);
    }
  }

  return out;
}

Matrix Matrix::operator*(Matrix rhs) {
  if (rhs.cols != rows) {
    throw std::invalid_argument(
        "Matrices must have the same inner dimensions (i.e. (a, b) * (b, c)");
  }

  Matrix out(rows, rhs.cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < rhs.cols; j++) {
      out(i, j) = 0;
      for (size_t k = 0; k < cols; k++) {
        out(i, j) += mat[i][k] * rhs(k, j);
      }
    }
  }

  return out;
}

Matrix Matrix::operator*(float rhs) {
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) = mat[i][j] * rhs;
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

Matrix Matrix::transpose() {
  Matrix out(cols, rows);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out(j, i) = mat[i][j];
    }
  }

  return out;
}

Matrix Matrix::hadamard_product(Matrix rhs) {
  if (rhs.rows != rows || rhs.cols != cols) {
    throw std::invalid_argument(
        "Matrices must be the same size to take Hadamard product.");
  }

  Matrix out(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out(i, j) = mat[i][j] * rhs(i, j);
    }
  }

  return out;
}

size_t Matrix::get_rows() { return rows; }

size_t Matrix::get_cols() { return cols; }
