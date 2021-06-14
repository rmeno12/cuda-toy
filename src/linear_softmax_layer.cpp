#include "linear_softmax_layer.hpp"

#include <random>

LinearSoftmaxLayer::LinearSoftmaxLayer(size_t input_size, size_t size)
    : LinearLayer(input_size, size) {
  initialize_weights();
}

// Using Xavier initialization
void LinearSoftmaxLayer::initialize_weights() {
  std::default_random_engine gen;
  std::normal_distribution<float> dist(0.0, sqrtf32((size + input_size) / 2.0));

  for (auto i = 0; i < size; i++) {
    for (auto j = 0; j < input_size; j++) {
      weights(i, j) = dist(gen);
    }
  }
}

Matrix LinearSoftmaxLayer::activate(Matrix input) {
  float sum = 0;
  for (auto i = 0; i < input.get_rows(); i++) {
    sum += std::exp(input(i, 0));
  }

  Matrix out(input.get_rows(), 1);
  for (auto i = 0; i < input.get_rows(); i++) {
    out(i, 0) = std::exp(input(i, 0)) / sum;
  }

  return out;
}

Matrix LinearSoftmaxLayer::backprop(Matrix activations, Matrix truths) {
  error = activations - truths;

  return error;
}