#include "matrix.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
  mat = new float*[rows];
  mat[0] = new float[rows * cols];
  for (auto i = 0; i < rows; i++) {
    if (i > 0) {
      mat[i] = mat[i - 1] + cols;
    }
    for (auto j = 0; j < cols; j++) {
      mat[i][j] = 0;
    }
  }
}

Matrix::Matrix(const Matrix& rhs) {
  rows = rhs.rows;
  cols = rhs.cols;
  mat = new float*[rows];
  mat[0] = new float[rows * cols];
  for (auto i = 0; i < rows; i++) {
    if (i > 0) {
      mat[i] = mat[i - 1] + cols;
    }
    for (auto j = 0; j < cols; j++) {
      mat[i][j] = rhs.mat[i][j];
    }
  }
}

Matrix::Matrix(std::vector<float> vals, size_t rows, size_t cols)
    : rows(rows), cols(cols) {
  if (rows * cols != vals.size()) {
    throw std::invalid_argument(
        "vals array must have same number of elements as desired matrix.");
  }

  mat = new float*[rows];
  mat[0] = new float[rows * cols];
  for (auto i = 0; i < rows; i++) {
    if (i > 0) mat[i] = mat[i - 1] + cols;
    for (auto j = 0; j < cols; j++) {
      mat[i][j] = vals[i * cols + j];
    }
  }
}

Matrix::Matrix(float** vals, size_t rows, size_t cols)
    : rows(rows), cols(cols) {
  mat = new float*[rows];
  mat[0] = new float[rows * cols];
  for (auto i = 0; i < rows; i++) {
    if (i > 0) mat[i] = mat[i - 1] + cols;
    for (auto j = 0; j < cols; j++) {
      mat[i][j] = vals[i][j];
    }
  }
}

Matrix::~Matrix() {
  delete[] mat[0];
  delete[] mat;
}

Matrix& Matrix::operator=(const Matrix& rhs) {
  if (this == &rhs) return *this;

  delete[] mat[0];
  delete[] mat;

  rows = rhs.rows;
  cols = rhs.cols;
  mat = new float*[rows];
  mat[0] = new float[rows * cols];
  for (auto i = 0; i < rows; i++) {
    if (i > 0) {
      mat[i] = mat[i - 1] + cols;
    }
    for (auto j = 0; j < cols; j++) {
      mat[i][j] = rhs.mat[i][j];
    }
  }

  return *this;
}

void Matrix::augment(const Matrix& other, int axis) {
  if (axis != 0 && axis != 1)
    throw std::invalid_argument("Axis must be either 0 or 1");

  float** newmat;
  if (axis == 0) {
    if (other.cols != cols)
      throw std::invalid_argument(
          "To augment on axis 0, both matrices must have the same number of "
          "columns");
    rows += other.rows;
    newmat = new float*[rows];
    newmat[0] = new float[rows * cols];
    for (auto i = 0; i < rows; i++) {
      if (i > 0) newmat[i] = newmat[i - 1] + cols;
      for (auto j = 0; j < cols; j++) {
        if (i < rows - other.rows) {
          newmat[i][j] = mat[i][j];
        } else {
          newmat[i][j] = other.mat[rows - i - 1][j];
        }
      }
    }
    delete[] mat[0];
    delete[] mat;
    mat = newmat;
  } else {
    if (other.rows != rows)
      throw std::invalid_argument(
          "To augment on axis 1, both matrices must have the same number of "
          "rows");
    cols += other.cols;
    newmat = new float*[rows];
    newmat[0] = new float[rows * cols];
    for (auto i = 0; i < rows; i++) {
      if (i > 0) newmat[i] = newmat[i - 1] + cols;
      for (auto j = 0; j < cols; j++) {
        if (j < cols - other.cols) {
          newmat[i][j] = mat[i][j];
        } else {
          newmat[i][j] = other.mat[i][cols - j - 1];
        }
      }
    }
    delete[] mat[0];
    delete[] mat;
    mat = newmat;
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

  matadd_wrapper(mat[0], rhs.mat[0], rows, cols, broadcast);

  return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
  if (rhs.rows != rows) {
    throw std::invalid_argument(
        "Matrices must either be the same size to be subtracted together.");
  }

  bool broadcast = rhs.cols == 1;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] -= broadcast ? rhs(i, 0) : rhs(i, j);
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

  float** result = new float*[rows];
  result[0] = new float[rows * rhs.cols];
  for (auto i = 1; i < rows; i++) result[i] = result[i - 1] + rhs.cols;

  matmul_wrapper(mat[0], rhs.mat[0], result[0], rows, cols, rhs.cols);
  Matrix out(result, rows, rhs.cols);

  delete[] result[0];
  delete[] result;

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

Matrix& Matrix::operator+=(float rhs) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] += rhs;
    }
  }

  return *this;
}

Matrix& Matrix::operator-=(float rhs) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] -= rhs;
    }
  }

  return *this;
}

Matrix& Matrix::operator*=(float rhs) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] *= rhs;
    }
  }

  return *this;
}

Matrix& Matrix::operator/=(float rhs) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat[i][j] /= rhs;
    }
  }

  return *this;
}

const Matrix Matrix::operator+(float rhs) const {
  Matrix result = *this;
  result += rhs;
  return result;
}

const Matrix Matrix::operator-(float rhs) const {
  Matrix result = *this;
  result -= rhs;
  return result;
}

const Matrix Matrix::operator*(float rhs) const {
  Matrix result = *this;
  result *= rhs;
  return result;
}

const Matrix Matrix::operator/(float rhs) const {
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

const Matrix Matrix::mean(int axis) const {
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

const Matrix Matrix::sum(int axis) const {
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

const Matrix Matrix::max(int axis) const {
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
      float big = mat[0][j];
      for (auto i = 0; i < rows; i++) {
        big = std::max(big, mat[i][j]);
      }
      out(0, j) = big;
    }
  } else {
    for (auto i = 0; i < rows; i++) {
      float big = mat[i][0];
      for (auto j = 0; j < cols; j++) {
        big = std::max(big, mat[i][j]);
      }
      out(i, 0) = big;
    }
  }

  return out;
}

const Matrix Matrix::maximum(const Matrix& lhs, float rhs) {
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

float Matrix::sum(const Matrix& input) {
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

size_t Matrix::get_rows() const { return rows; }

size_t Matrix::get_cols() const { return cols; }

const Matrix operator+(float lhs, const Matrix& rhs) { return rhs + lhs; }

const Matrix operator-(float lhs, const Matrix& rhs) { return -rhs + lhs; }

const Matrix operator*(float lhs, const Matrix& rhs) { return rhs * lhs; }

const Matrix operator/(float lhs, const Matrix& rhs) {
  Matrix out(rhs.get_rows(), rhs.get_cols());

  for (auto i = 0; i < rhs.get_rows(); i++) {
    for (auto j = 0; j < rhs.get_cols(); j++) {
      out(i, j) = lhs / rhs(i, j);
    }
  }

  return out;
}