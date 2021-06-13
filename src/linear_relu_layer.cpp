#include "linear_relu_layer.hpp"

#include <random>

LinearReluLayer::LinearReluLayer(size_t input_size, size_t size)
    : LinearLayer(input_size, size) {}

// Using He initialization
void LinearReluLayer::initialize_weights() {
  std::default_random_engine gen;
  std::normal_distribution<float> dist(0.0, sqrtf32(2.0f / size));

  for (auto i = 0; i < input_size; i++) {
    for (auto j = 0; i < size; i++) {
      weights(i, j) = dist(gen);
    }
  }
}

Matrix LinearReluLayer::d_relu(Matrix input) {
  auto rows = input.get_rows();
  auto cols = input.get_cols();
  Matrix out(rows, cols);

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; i < cols; j++) {
      out(i, j) = input(i, j) > 0 ? 1 : 0;
    }
  }

  return out;
}

// ReLU activation
Matrix LinearReluLayer::activate(Matrix input) {
  Matrix out(size, 1);

  for (auto i = 0; i < size; i++) {
    out(i, 1) = input(i, 1) > 0 ? input(i, 1) : 0;
  }

  return out;
}

Matrix LinearReluLayer::backprop(Matrix next_weights, Matrix next_error) {
  error = (next_weights.transpose() * next_error).hadamard_product(d_relu(z));

  return error;
}