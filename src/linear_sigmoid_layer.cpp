#include "linear_sigmoid_layer.hpp"

#include <random>

LinearSigmoidLayer::LinearSigmoidLayer(const size_t& input_size,
                                       const size_t& output_size)
    : LinearLayer(input_size, output_size) {
  init_weights();
  init_biases();
}

void LinearSigmoidLayer::init_weights() {
  std::default_random_engine eng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (auto i = 0; i < output_size; i++) {
    for (auto j = 0; j < input_size; j++) {
      weights(i, j) =
          dist(eng) / sqrtf32((float)(input_size + output_size) / 2.0);
    }
  }
}

void LinearSigmoidLayer::init_biases() {
  for (auto i = 0; i < output_size; i++) {
    biases(i, 0) = 0;
  }
}

const Matrix LinearSigmoidLayer::activate(const Matrix& input) const {
  return 1 / (1 + Matrix::exp(input));
}

const Matrix LinearSigmoidLayer::d_activate(const Matrix& input) const {
  Matrix sig = activate(input);
  return sig.product(-sig - 1);
}

const Matrix LinearSigmoidLayer::backward(const Matrix& m1, const Matrix& m2) {
  // TODO: non last layer version
  Matrix dl_da(1, 1);
  if (last_layer) {
    dl_da = (-m1 - 1).divide(-a - 1) - m1.divide(a);
  } else {
  }
  Matrix da_dz = d_activate(z);
  d_loss = dl_da.product(da_dz);

  return d_loss;
}