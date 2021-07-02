#include "linear_relu_layer.hpp"

#include <random>

LinearReluLayer::LinearReluLayer(const size_t& input_size,
                                 const size_t& output_size)
    : LinearLayer(input_size, output_size) {
  init_weights();
  init_biases();
}

void LinearReluLayer::init_weights() {
  std::default_random_engine eng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (auto i = 0; i < output_size; i++) {
    for (auto j = 0; j < input_size; j++) {
      weights(i, j) =
          dist(eng) / sqrtf32((float)(input_size + output_size) / 2.0);
    }
  }
}

void LinearReluLayer::init_biases() {
  for (auto i = 0; i < output_size; i++) {
    biases(i, 0) = 0;
  }
}

const Matrix LinearReluLayer::activate(const Matrix& input) const {
  return Matrix::maximum(input, 0);
}

const Matrix LinearReluLayer::d_activate(const Matrix& input) const {
  Matrix out(input.get_rows(), input.get_cols());

  for (auto i = 0; i < input.get_rows(); i++) {
    for (auto j = 0; j < input.get_cols(); j++) {
      out(i, j) = input(i, j) > 0 ? 1.0 : 0.0;
    }
  }

  return out;
}

const Matrix LinearReluLayer::backward(const Matrix& m1, const Matrix& m2) {
  // TODO: last layer version
  Matrix dl_da(1, 1);
  if (last_layer) {
  } else {
    dl_da = m1.transpose() * m2;
  }
  Matrix da_dz = d_activate(z);
  d_loss = dl_da.product(da_dz);

  return d_loss;
}