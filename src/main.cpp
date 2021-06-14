#include <math.h>

#include <iostream>

#include "linear_sigmoid_layer.hpp"
#include "linear_softmax_layer.hpp"
#include "mnist.hpp"
#include "model.hpp"

void test();
void backprop(Matrix m1, Matrix m2);
void update();

int main() {
  // load the data
  // Mnist mnist("/workspaces/cuda-toy/data");
  std::vector<Matrix> training_data = {
      Matrix({1, 0}, 2, 1),
      Matrix({1, 1}, 2, 1),
      Matrix({0, 0}, 2, 1),
      Matrix({0, 1}, 2, 1),
  };
  std::vector<Matrix> training_labels = {
      Matrix({1, 0}, 2, 1),
      Matrix({0, 1}, 2, 1),
      Matrix({0, 1}, 2, 1),
      Matrix({1, 0}, 2, 1),
  };

  float learning_rate = 0.05;
  Model model(learning_rate);

  int iters = 50;
  for (int i = 0; i < iters; i++) {
    if (i % 1 == 0) {
      // float avg = 0;
      // for (int j = 0; j < training_data.size(); j++) {
      //   avg += model.rmse(training_data[j], training_labels[j]) /
      //          training_data.size();
      // }
      float avg = model.rmse(training_data[1], training_labels[1]);
      std::cout << "iter " << i << " cost " << avg << std::endl;
    }
    // for (int j = 0; j < training_data.size(); j++) {
    //   model.train(training_data[j], training_labels[j]);
    // }
    model.train(training_data[1], training_labels[1]);
  }
  // test();
}

std::vector<LinearLayer*> layers;

void test() {
  LinearSigmoidLayer l1(2, 2);
  LinearSoftmaxLayer l2(2, 2);
  layers = {&l1, &l2};
  // l1.get_weights().print();
  // l2.get_weights().print();
  // std::cout << std::endl;

  Matrix input({1, 0}, 2, 1);
  Matrix truth({0, 1}, 2, 1);
  // input.print();
  Matrix ff1 = l1.feed_forward(input);
  ff1.print();
  Matrix ff2 = l2.feed_forward(ff1);
  ff2.print();

  for (auto i = 0; i < 50; i++) {
    Matrix prev = input;
    for (auto layer : layers) {
      prev = layer->feed_forward(prev);
    }

    Matrix m1 = prev;
    Matrix m2 = truth;
    backprop(m1, m2);

    update();

    Matrix errs = (prev - truth).hadamard_product(prev - truth);
    float sum = 0;
    for (auto i = 0; i < errs.get_rows(); i++) {
      sum += errs(i, 0) / errs.get_rows();
    }
    std::cout << "err " << sqrt(sum) << std::endl;
  }

  ff1 = l1.feed_forward(input);
  ff1.print();
  ff2 = l2.feed_forward(ff1);
  ff2.print();
}

void backprop(Matrix m1, Matrix m2) {
  for (int i = layers.size() - 1; i >= 0; i--) {
    layers[i]->backprop(m1, m2);
    m1 = layers[i]->get_weights();
    m2 = layers[i]->get_error();
  }
}

void update() {
  for (auto layer : layers) {
    layer->update_params(0.05);
  }
}