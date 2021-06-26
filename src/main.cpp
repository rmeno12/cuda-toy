#include "linear_relu_layer.hpp"
#include "linear_sigmoid_layer.hpp"
#include "matrix.hpp"

Matrix X({1, 1, 0, 0, 1, 0, 0, 1}, 2, 4);
Matrix Y({0, 1, 0, 1}, 1, 4);

float learning_rate = 0.03;

int x_size = 2;
int l1_size = 10;
int l2_size = 1;
LinearReluLayer l1(x_size, l1_size);
LinearSigmoidLayer l2(l1_size, l2_size);

Matrix forward_backward(Matrix x, Matrix y) {
  // forward pass
  Matrix a1 = l1.forward(x);
  Matrix a2 = l2.forward(a1);

  Matrix loss =
      (-y).product(Matrix::log2(a2)) - (-y - 1).product(Matrix::log2(-a2 - 1));

  // backward pass
  Matrix dl_dz2 = l2.backward(y, a2);
  Matrix dl_dz1 = l1.backward(l2.get_weights(), dl_dz2);

  Matrix dl_dw2 = dl_dz2 * a1.transpose();
  Matrix dl_db2 = dl_dz2.mean(1);
  Matrix dl_dw1 = dl_dz1 * x.transpose();
  Matrix dl_db1 = dl_dz1.mean(1);

  return loss;
}

Matrix predict(Matrix x) {
  Matrix a1 = l1.forward(x);
  Matrix a2 = l2.forward(a1);

  return a2;
}

int main() {
  l2.set_last_layer(true);
  predict(X).print();
  std::vector<float> losses;

  for (auto i = 0; i < 10000; i++) {
    Matrix x = X;
    Matrix y = Y;
    Matrix loss = forward_backward(x, y);

    l1.update_params(learning_rate);
    l2.update_params(learning_rate);

    losses.push_back(loss.mean(0)(0, 0));
  }

  predict(X).print();
}