#include "matrix.hpp"

#include <cmath>
#include <iostream>
#include <limits>
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

Matrix::Matrix(std::vector<float> vals, size_t rows, size_t cols)
    : rows(rows), cols(cols) {
  if (rows * cols != vals.size()) {
    throw std::invalid_argument(
        "vals array must have same number of elements as desired matrix.");
  }

  mat.resize(rows);
  for (auto i = 0; i < rows; i++) {
    mat[i] = {};
    mat[i].resize(cols);
  }

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      mat[i][j] = vals[i * cols + j];
    }
  }
}

Matrix& Matrix::operator=(const Matrix& rhs) {
  if (this == &rhs) return *this;

  rows = rhs.rows;
  cols = rhs.cols;
  mat.resize(rows);
  for (auto i = 0; i < rows; i++) {
    mat[i] = {};
    mat[i].resize(cols);
  }

  for (size_t i = 0; i < rhs.rows; i++) {
    for (size_t j = 0; j < rhs.cols; j++) {
      mat[i][j] = rhs(i, j);
    }
  }

  return *this;
}

float& Matrix::operator()(size_t i, size_t j) {
  if (i >= rows || j >= cols) {
    throw std::out_of_range("Trying to access matrix out of bounds");
  }

  return mat[i][j];
}

float Matrix::operator()(size_t i, size_t j) const {
  if (i >= rows || j >= cols) {
    throw std::out_of_range("Trying to access matrix out of bounds");
  }

  return mat[i][j];
}

Matrix Matrix::operator+(Matrix rhs) {
  if (rhs.rows != rows) {
    throw std::invalid_argument(
        "Matrices must either be the same size or have the same rows and the "
        "second matrix have only one column to be added together.");
  }

  // TODO: make this more efficient
  bool expand = false;
  if (rhs.cols != cols && rhs.cols == 1) {
    expand = true;
  } else if (rhs.cols != cols && rhs.cols != 1) {
    throw;
  }

  Matrix out(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out(i, j) = mat[i][j] + rhs(i, expand ? 0 : j);
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
  if (cols != rhs.rows) {
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

Matrix Matrix::operator+(float rhs) {
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) = mat[i][j] - rhs;
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

Matrix Matrix::operator-() {
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) = -mat[i][j];
    }
  }

  return out;
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

Matrix Matrix::product(Matrix rhs) {
  if (rhs.rows != rows || rhs.cols != cols) {
    throw std::invalid_argument(
        "Matrices must be the same size to perform elementwise "
        "multiplication.");
  }

  Matrix out(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out(i, j) = mat[i][j] * rhs(i, j);
    }
  }

  return out;
}

Matrix Matrix::divide(Matrix rhs) {
  if (rhs.rows != rows || rhs.cols != cols) {
    throw std::invalid_argument(
        "Matrices must be the same size to perform elementwise division.");
  }

  Matrix out(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out(i, j) = mat[i][j] / rhs(i, j);
    }
  }

  return out;
}

Matrix Matrix::mean(int axis) {
  Matrix* tmp;
  if (axis == 0) {
    tmp = new Matrix(1, cols);
  } else if (axis == 1) {
    tmp = new Matrix(rows, 1);
  } else {
    throw std::invalid_argument("axis must be 0 or 1.");
  }
  Matrix out = *tmp;
  delete tmp;

  if (axis == 0) {
    for (auto j = 0; j < cols; j++) {
      float avg = 0;
      for (auto i = 0; i < rows; i++) {
        avg += mat[i][j] / rows;
      }
      out(0, j) = avg;
    }
  } else {
    for (auto i = 0; i < rows; i++) {
      float avg = 0;
      for (auto j = 0; j < cols; j++) {
        avg += mat[i][j] / cols;
      }
      out(i, 0) = avg;
    }
  }

  return out;
}

Matrix Matrix::maximum(Matrix lhs, float rhs) {
  int rows = lhs.get_rows();
  int cols = lhs.get_cols();
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) = std::max(lhs(i, j), rhs);
    }
  }

  return out;
}

Matrix Matrix::exp(Matrix input) {
  int rows = input.get_rows();
  int cols = input.get_cols();
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) = std::exp(out(i, j));
    }
  }

  return out;
}

Matrix Matrix::log2(Matrix input) {
  int rows = input.get_rows();
  int cols = input.get_cols();
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) = std::log2f(out(i, j));
    }
  }

  return out;
}

void Matrix::print() {
  std::cout << "{ ";
  for (auto i = 0; i < rows; i++) {
    if (i > 0) {
      std::cout << "  ";
    }
    std::cout << "{";
    for (auto j = 0; j < cols; j++) {
      std::cout << " " << mat[i][j];
    }
    std::cout << " }";
    if (i + 1 < rows) {
      std::cout << "\n";
    }
  }
  std::cout << " }\n\n";
}

size_t Matrix::get_rows() { return rows; }

size_t Matrix::get_cols() { return cols; }
