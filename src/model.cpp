#include "model.hpp"

#include <math.h>

#include "linear_relu_layer.hpp"
#include "linear_sigmoid_layer.hpp"
#include "linear_softmax_layer.hpp"

Model::Model(float learning_rate = 0.01) : learning_rate(learning_rate) {
  layers = {};
  // layers.push_back(new LinearReluLayer(784, 128));
  // layers.push_back(new LinearReluLayer(128, 64));
  // layers.push_back(new LinearSoftmaxLayer(64, 10));
  layers.push_back(new LinearSigmoidLayer(2, 2));
  layers.push_back(new LinearSigmoidLayer(2, 1));
  // layers.push_back(new LinearSoftmaxLayer(2, 2));
  layers.back()->set_is_last(true);
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

Matrix Model::matrix_log(Matrix input) {
  Matrix out(input.get_rows(), input.get_cols());

  for (auto i = 0; i < input.get_rows(); i++) {
    for (auto j = 0; j < input.get_cols(); j++) {
      out(i, j) = std::log2(input(i, j));
    }
  }

  return out;
}

void Model::backprop(Matrix preds, Matrix truths) {
  Matrix m1 = preds;
  Matrix m2 = truths;
  for (int i = layers.size() - 1; i >= 0; i--) {
    m2 = layers[i]->backprop(m1, m2);
    m1 = layers[i]->get_weights();
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

  return prev;
}

float Model::rmse(Matrix input, Matrix truths) {
  Matrix preds = predict(input);
  // Matrix errs = (truths).hadamard_product(matrix_log(preds)) * -1.0;
  // float sum = 0;
  // for (auto i = 0; i < errs.get_rows(); i++) {
  //   sum += errs(i, 0);
  // }

  // return sum;

  Matrix errs = (truths - preds).hadamard_product(truths - preds);

  float sum = 0;
  for (auto i = 0; i < errs.get_rows(); i++) {
    sum += errs(i, 0) / errs.get_rows();
  }

  return sqrt(sum);
}

void Model::train(Matrix input, Matrix truths) {
  Matrix prev = input;

  for (auto layer : layers) {
    prev = layer->feed_forward(prev);
  }

  backprop(prev, truths);

  update_params();
}

void Model::train_batch(Matrix input, Matrix truths) {
  Matrix prev = input;

  for (auto layer : layers) {
    prev = layer->feed_forward(prev);
  }

  backprop(prev, truths);

  update_params();
}