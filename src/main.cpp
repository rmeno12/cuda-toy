#include <random>

#include "linear_relu_layer.hpp"
#include "matrix.hpp"

Matrix sigmoid(Matrix input) {
  Matrix out(input.get_rows(), input.get_cols());

  for (auto i = 0; i < input.get_rows(); i++) {
    for (auto j = 0; j < input.get_cols(); j++) {
      out(i, j) = 1.0 / (1.0 + std::exp(input(i, j)));
    }
  }

  return out;
}

Matrix d_sigmoid(Matrix input) {
  Matrix sig = sigmoid(input);
  return sig.product(-sig - 1.0);
}

Matrix relu(Matrix input) { return Matrix::maximum(input, 0); }

Matrix d_relu(Matrix input) {
  Matrix out(input.get_rows(), input.get_cols());

  for (auto i = 0; i < input.get_rows(); i++) {
    for (auto j = 0; j < input.get_cols(); j++) {
      out(i, j) = out(i, j) > 0 ? 1.0 : 0.0;
    }
  }

  return out;
}

Matrix init_weights(size_t in_size, size_t out_size) {
  std::default_random_engine eng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  Matrix out(out_size, in_size);

  for (auto i = 0; i < out_size; i++) {
    for (auto j = 0; j < in_size; j++) {
      out(i, j) = dist(eng) / sqrtf32((float)(in_size + out_size) / 2.0);
    }
  }

  return out;
}

Matrix X({1, 1, 0, 0, 1, 0, 0, 1}, 2, 4);
Matrix Y({0, 1, 0, 1}, 1, 4);

float learning_rate = 0.03;

int x_size = 2;
int l1_size = 10;
int l2_size = 1;
LinearReluLayer l1(x_size, l1_size);
Matrix w2 = init_weights(l1_size, l2_size);
Matrix b2 = Matrix(l2_size, 1);

Matrix* forward_backward(Matrix x, Matrix y) {
  // forward pass
  Matrix a1 = l1.forward(x);
  Matrix z2 = w2 * a1 + b2;
  Matrix a2 = sigmoid(z2);

  Matrix loss =
      (-y).product(Matrix::log2(a2)) - (-y - 1).product(Matrix::log2(-a2 - 1));

  // backward pass
  Matrix dl_da2 = (-y - 1).divide(-a2 - 1) - y.divide(a2);
  Matrix da2_dz2 = d_sigmoid(z2);
  Matrix dl_dz2 = dl_da2.product(da2_dz2);

  Matrix dl_dz1 = l1.backward(w2, dl_dz2);

  Matrix dl_dw2 = dl_dz2 * a1.transpose();
  Matrix dl_db2 = dl_dz2.mean(1);
  Matrix dl_dw1 = dl_dz1 * x.transpose();
  Matrix dl_db1 = dl_dz1.mean(1);

  return new Matrix[5]{loss, dl_dw1, dl_db1, dl_dw2, dl_db2};
}

Matrix predict(Matrix x) {
  Matrix a1 = l1.forward(x);
  Matrix z2 = w2 * a1 + b2;
  Matrix a2 = sigmoid(z2);

  return a2;
}

int main() {
  predict(X).print();
  std::vector<float> losses;

  for (auto i = 0; i < 10000; i++) {
    Matrix x = X;
    Matrix y = Y;
    Matrix* results = forward_backward(x, y);

    l1.update_params(learning_rate);
    w2 = w2 - results[3] * learning_rate;
    b2 = b2 - results[4] * learning_rate;

    losses.push_back(results[0].mean(0)(0, 0));
  }

  predict(X).print();
}