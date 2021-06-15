#include "linear_relu_layer.hpp"

#include <random>

LinearReluLayer::LinearReluLayer(size_t input_size, size_t size)
    : LinearLayer(input_size, size) {
  initialize_weights();
}

// Using He initialization
void LinearReluLayer::initialize_weights() {
  std::default_random_engine gen;
  std::normal_distribution<float> dist(0.0, sqrtf32(2.0f / size));

  for (auto i = 0; i < size; i++) {
    for (auto j = 0; j < input_size; j++) {
      weights(i, j) = dist(gen);
    }
  }
}

Matrix LinearReluLayer::d_relu(Matrix input) {
  auto rows = input.get_rows();
  auto cols = input.get_cols();
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      out(i, j) = input(i, j) > 0 ? 1 : 0;
    }
  }

  return out;
}

// ReLU activation
Matrix LinearReluLayer::activate(Matrix input) {
  Matrix out(size, input.get_cols());

  for (auto i = 0; i < size; i++) {
    for (auto j = 0; j < input.get_cols(); j++) {
      out(i, j) = input(i, j) > 0 ? input(i, j) : 0;
    }
  }

  return out;
}

Matrix LinearReluLayer::backprop(Matrix next_weights, Matrix next_error) {
  error = (next_weights.transpose() * next_error).hadamard_product(d_relu(z));

  return error;
}