#include "model.hpp"

#include "linear_relu_layer.hpp"
#include "linear_softmax_layer.hpp"

Model::Model(float learning_rate = 0.01) : learning_rate(learning_rate) {
  layers = {};
  layers.push_back(new LinearReluLayer(784, 128));
  layers.push_back(new LinearReluLayer(128, 64));
  layers.push_back(new LinearSoftmaxLayer(64, 10));
}

Matrix Model::make_one_hot(Matrix input) {
  auto max = 0;
  auto max_idx = 0;
  for (auto i = 0; i < input.get_rows(); i++) {
    if (input(i, 0) > max) {
      max_idx = i;
    }
  }

  Matrix out(input.get_rows(), 1);
  for (auto i = 0; i < input.get_rows(); i++) {
    if (i == max_idx) {
      out(i, 0) = 1;
    } else {
      out(i, 0) = 0;
    }
  }

  return out;
}

void Model::backprop(Matrix preds, Matrix truths) {
  Matrix m1 = preds;
  Matrix m2 = truths;
  for (auto i = layers.size() - 1; i >= 0; i--) {
    layers[i]->backprop(m1, m2);
    m1 = layers[i]->get_weights();
    m2 = layers[i]->get_error();
  }
}

void Model::update_params() {
  for (auto layer : layers) {
    layer->update_params(learning_rate);
  }
}

Matrix Model::predict(Matrix input) {
  Matrix prev = input;

  for (auto layer : layers) {
    prev = layer->feed_forward(prev);
  }

  return make_one_hot(prev);
}

void Model::train(Matrix input, Matrix truths) {
  Matrix prev = input;

  for (auto layer : layers) {
    prev = layer->feed_forward(prev);
  }

  backprop(prev, truths);

  update_params();
}