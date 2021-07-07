#include "linear_softmax_layer.hpp"

#include <random>

LinearSoftmaxLayer::LinearSoftmaxLayer(const size_t& input_size,
                                       const size_t& output_size)
    : LinearLayer(input_size, output_size) {
  init_weights();
  init_biases();
}

void LinearSoftmaxLayer::init_weights() {
  std::default_random_engine eng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (auto i = 0; i < output_size; i++) {
    for (auto j = 0; j < input_size; j++) {
      weights(i, j) =
          dist(eng) / sqrtf32((float)(input_size + output_size) / 2.0);
    }
  }
}

void LinearSoftmaxLayer::init_biases() {
  for (auto i = 0; i < output_size; i++) {
    biases(i, 0) = 0;
  }
}

const Matrix LinearSoftmaxLayer::activate(const Matrix& input) const {
  Matrix out =
      Matrix::exp((input.transpose() - input.max(0).transpose()).transpose());
  for (auto j = 0; j < out.get_cols(); j++) {
    float sum = 0;
    for (auto i = 0; i < out.get_rows(); i++) {
      sum += out(i, j);
    }
    for (auto i = 0; i < out.get_rows(); i++) {
      out(i, j) /= std::max(std::numeric_limits<float>().min(), sum);
    }
  }

  return out;
}

const Matrix LinearSoftmaxLayer::d_activate(const Matrix& input) const {
  // This function is never used
  return input;
}

const Matrix LinearSoftmaxLayer::backward(const Matrix& m1, const Matrix& m2) {
  if (last_layer) {
    d_loss = a - m1;  // See math document for explanation of this
  } else {
    // Should never be used as last layer
  }

  return d_loss;
}