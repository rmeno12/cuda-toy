#include "mnist_model.hpp"

#include "linear_relu_layer.hpp"
#include "linear_softmax_layer.hpp"

MnistModel::MnistModel() {
  int x_size = 784;
  int l1_size = 128;
  int l2_size = 10;
  layers = {new LinearReluLayer(x_size, l1_size),
            new LinearSoftmaxLayer(l1_size, l2_size)};
  layers.back()->set_last_layer(true);
  learning_rate = 0.005;
}

const Matrix MnistModel::predict(const Matrix& x) {
  Matrix pred = x;
  for (auto layer : layers) {
    pred = layer->forward(pred);
  }

  return pred;
}

const Matrix MnistModel::train(const Matrix& x, const Matrix& y) {
  Matrix yhat = predict(x);

  Matrix loss = -(y.product(Matrix::log2(yhat)).sum(0));

  Matrix m1 = y;
  Matrix m2 = yhat;
  for (int i = layers.size() - 1; i >= 0; i--) {
    m2 = layers[i]->backward(m1, m2);
    m1 = layers[i]->get_weights();
  }

  for (auto layer : layers) {
    layer->update_params(learning_rate);
  }

  return loss;
}