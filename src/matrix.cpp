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

void Matrix::augment(const Matrix& other, int axis) {
  if (axis != 0 && axis != 1)
    throw std::invalid_argument("Axis must be either 0 or 1");

  if (axis == 0) {
    if (other.cols != cols)
      throw std::invalid_argument(
          "To augment on axis 0, both matrices must have the same number of "
          "columns");
    rows += other.rows;
    for (auto row : other.mat) {
      mat.push_back(row);
    }
  } else {
    if (other.rows != rows)
      throw std::invalid_argument(
          "To augment on axis 1, both matrices must have the same number of "
          "rows");
    cols += other.cols;
    for (auto i = 0; i < rows; i++) {
      for (auto j = 0; j < other.cols; j++) {
        mat[i].push_back(other.mat[i][j]);
      }
    }
  }
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

const Matrix Matrix::operator-() const {
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) = -mat[i][j];
    }
  }

  return out;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
  if (rhs.rows != rows) {
    throw std::invalid_argument(
        "Matrices must be the same size to be added together.");
  }

  bool broadcast = rhs.cols == 1;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] += broadcast ? rhs(i, 0) : rhs(i, j);
    }
  }

  return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
  if (rhs.rows != rows) {
    throw std::invalid_argument(
        "Matrices must either be the same size to be subtracted together.");
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] -= rhs(i, j);
    }
  }

  return *this;
}

const Matrix Matrix::operator+(const Matrix& rhs) const {
  Matrix result = *this;
  result += rhs;
  return result;
}

const Matrix Matrix::operator-(const Matrix& rhs) const {
  Matrix result = *this;
  result -= rhs;
  return result;
}

const Matrix Matrix::operator*(const Matrix& rhs) const {
  if (cols != rhs.rows) {
    throw std::invalid_argument(
        "Matrices must have the same inner dimensions (i.e. (a, b) * (b, c)");
  }

  Matrix out(rows, rhs.cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < rhs.cols; j++) {
      out(i, j) = 0;
      for (size_t k = 0; k < cols; k++) {
        float x = mat[i][k];
        float y = rhs(k, j);
        float to_add = mat[i][k] * rhs(k, j);
        if (std::isnan(to_add)) {
          throw;
        }
        out(i, j) += mat[i][k] * rhs(k, j);
      }
    }
  }

  return out;
}

const Matrix Matrix::product(const Matrix& rhs) const {
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

const Matrix Matrix::divide(const Matrix& rhs) const {
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

Matrix& Matrix::operator+=(const float& rhs) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] += rhs;
    }
  }

  return *this;
}

Matrix& Matrix::operator-=(const float& rhs) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] -= rhs;
    }
  }

  return *this;
}

Matrix& Matrix::operator*=(const float& rhs) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] *= rhs;
    }
  }

  return *this;
}

Matrix& Matrix::operator/=(const float& rhs) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] /= rhs;
    }
  }

  return *this;
}

const Matrix Matrix::operator+(const float& rhs) const {
  Matrix result = *this;
  result += rhs;
  return result;
}

const Matrix Matrix::operator-(const float& rhs) const {
  Matrix result = *this;
  result -= rhs;
  return result;
}

const Matrix Matrix::operator*(const float& rhs) const {
  Matrix result = *this;
  result *= rhs;
  return result;
}

const Matrix Matrix::operator/(const float& rhs) const {
  Matrix result = *this;
  result /= rhs;
  return result;
}

const Matrix Matrix::transpose() const {
  Matrix out(cols, rows);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out(j, i) = mat[i][j];
    }
  }

  return out;
}

const Matrix Matrix::mean(const int& axis) const {
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

const Matrix Matrix::sum(const int& axis) const {
  Matrix* tmp;
  if (axis == 0) {
    tmp = new Matrix(1, cols);
  } else if (axis == 1) {
    tmp = new Matrix(rows, 1);
  } else {
    throw std::invalid_argument("Axis must be 0 or 1.");
  }
  Matrix out = *tmp;
  delete tmp;

  if (axis == 0) {
    for (auto j = 0; j < cols; j++) {
      float sum = 0;
      for (auto i = 0; i < rows; i++) {
        sum += mat[i][j];
      }
      out(0, j) = sum;
    }
  } else {
    for (auto i = 0; i < rows; i++) {
      float sum = 0;
      for (auto j = 0; j < cols; j++) {
        sum += mat[i][j];
      }
      out(i, 0) = sum;
    }
  }

  return out;
}

const Matrix Matrix::maximum(const Matrix& lhs, const float& rhs) {
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

const Matrix Matrix::exp(const Matrix& input) {
  int rows = input.get_rows();
  int cols = input.get_cols();
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) =
          std::min(std::numeric_limits<float>().max(), std::exp(input(i, j)));
    }
  }

  return out;
}

const Matrix Matrix::log2(const Matrix& input) {
  int rows = input.get_rows();
  int cols = input.get_cols();
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) =
          std::log2f(std::max(std::numeric_limits<float>().min(), input(i, j)));
    }
  }

  return out;
}

const float Matrix::sum(const Matrix& input) {
  int rows = input.get_rows();
  int cols = input.get_cols();
  float sum = 0;

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      sum += input(i, j);
    }
  }

  return sum;
}

void Matrix::print() const {
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

const size_t Matrix::get_rows() const { return rows; }

const size_t Matrix::get_cols() const { return cols; }

const Matrix operator+(const float& lhs, const Matrix& rhs) {
  return rhs + lhs;
}

const Matrix operator-(const float& lhs, const Matrix& rhs) {
  return -rhs + lhs;
}

const Matrix operator*(const float& lhs, const Matrix& rhs) {
  return rhs * lhs;
}

const Matrix operator/(const float& lhs, const Matrix& rhs) {
  Matrix out(rhs.get_rows(), rhs.get_cols());

  for (auto i = 0; i < rhs.get_rows(); i++) {
    for (auto j = 0; j < rhs.get_cols(); j++) {
      out(i, j) = lhs / rhs(i, j);
    }
  }

  return out;
}