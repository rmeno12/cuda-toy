#include "linear_sigmoid_layer.hpp"

#include <random>

LinearSigmoidLayer::LinearSigmoidLayer(size_t input_size, size_t size)
    : LinearLayer(input_size, size) {
  initialize_weights();
}

// Using Xavier initialization
void LinearSigmoidLayer::initialize_weights() {
  std::default_random_engine gen;
  std::normal_distribution<float> dist(0.0, sqrtf32((size + input_size) / 2.0));

  for (auto i = 0; i < size; i++) {
    for (auto j = 0; j < input_size; j++) {
      weights(i, j) = dist(gen);
    }
  }
}

Matrix LinearSigmoidLayer::d_sigmoid(Matrix input) {
  return activate(input).hadamard_product(activate(input)) * -1.0;
}

// Sigmoid activation
Matrix LinearSigmoidLayer::activate(Matrix input) {
  Matrix out(input.get_rows(), input.get_cols());
  for (auto i = 0; i < input.get_rows(); i++) {
    for (auto j = 0; j < input.get_cols(); j++) {
      out(i, j) = 1.0 / (1 + std::exp(input(i, j)));
    }
  }

  return out;
}

Matrix LinearSigmoidLayer::backprop(Matrix m1, Matrix m2) {
  if (is_last_layer) {
    error = ((m2 + -1).hadamard_quotient(m1 + -1) - m2.hadamard_quotient(m1))
                .hadamard_product(d_sigmoid(z)) *
            LOG2E;
  } else {
    error = (m1.transpose() * m2).hadamard_product(d_sigmoid(z));
  }

  return error;
}