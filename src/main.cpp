#include <math.h>

#include <fstream>
#include <iostream>

#include "linear_sigmoid_layer.hpp"
#include "linear_softmax_layer.hpp"
#include "mnist.hpp"
#include "model.hpp"

void single_sample();
void stochastic();
void batch();
void test();
void backprop(Matrix m1, Matrix m2);
void update();

std::vector<Matrix> training_data = {
    Matrix({1, 0}, 2, 1),
    Matrix({1, 1}, 2, 1),
    Matrix({0, 0}, 2, 1),
    Matrix({0, 1}, 2, 1),
    Matrix({1, 1, 0, 0, 0, 1, 0, 1}, 2, 4),
};
std::vector<Matrix> training_labels = {
    Matrix({1}, 1, 1), Matrix({0}, 1, 1),          Matrix({0}, 1, 1),
    Matrix({1}, 1, 1), Matrix({1, 0, 0, 1}, 1, 4),
};

int main() {
  // load the data
  // Mnist mnist("/workspaces/cuda-toy/data");
  // std::vector<Matrix> training_data = {
  //     Matrix({1, 0}, 2, 1), Matrix({1, 1}, 2, 1), Matrix({0, 0}, 2, 1),
  //     Matrix({0, 1}, 2, 1), Matrix({1, 1, 0, 0, 0, 1, 0, 1}, 2, 4)};
  // std::vector<Matrix> training_labels = {
  //     Matrix({1, 0}, 2, 1), Matrix({0, 1}, 2, 1), Matrix({0, 1}, 2, 1),
  //     Matrix({1, 0}, 2, 1), Matrix({1, 0, 0, 1, 0, 1, 1, 0}, 2, 4)};

  // single_sample();
  // stochastic();
  batch();
  // test();
}

void single_sample() {
  float learning_rate = 0.5;
  Model model(learning_rate);

  int iters = 500;
  for (int i = 0; i < iters; i++) {
    if (i % 1 == 0) {
      float avg = model.rmse(training_data[1], training_labels[1]);
      std::cout << "iter " << i << " cost " << avg << std::endl;
    }
    model.train(training_data[1], training_labels[1]);
  }

  for (auto i = 0; i < 4; i++) {
    model.predict(training_data[i]).print();
  }
}

void stochastic() {
  float learning_rate = 0.5;
  Model model(learning_rate);

  int iters = 500;
  for (int i = 0; i < iters; i++) {
    float avg = 0;
    for (int j = 0; j < training_data.size(); j++) {
      avg += model.rmse(training_data[j], training_labels[j]) /
             training_data.size();
    }
    std::cout << "iter " << i << " cost " << avg << std::endl;
    for (int j = 0; j < training_data.size(); j++) {
      model.train(training_data[j], training_labels[j]);
    }
  }

  for (auto i = 0; i < 4; i++) {
    model.predict(training_data[i]).print();
  }
}

void batch() {
  float learning_rate = 0.05;
  Model model(learning_rate);

  for (auto i = 0; i < 4; i++) {
    model.predict(training_data[i]).print();
  }

  int iters = 500;
  for (int i = 0; i < iters; i++) {
    if (i % 10 == 0) {
      float avg = 0;
      for (int j = 0; j < 4; j++) {
        // avg += model.rmse(training_data[j], training_labels[j]) / 4;
        Matrix preds = model.predict(training_data[j]);
        Matrix err =
            (training_labels[j] * -1.0) * (Model::matrix_log(preds)) -
            ((training_labels[j] - Matrix({1}, 1, 1)) * -1.0) *
                (Model::matrix_log((preds - Matrix({1}, 1, 1)) * -1.0));
        avg += err(0, 0) / 4;
      }
      std::cout << "iter " << i << " cost " << avg << std::endl;
    }
    model.train_batch(training_data[4], training_labels[4]);
  }

  for (auto i = 0; i < 4; i++) {
    model.predict(training_data[i]).print();
  }
}

std::vector<LinearLayer*> layers;

void test() {
  std::ofstream outfile;
  outfile.open("output.csv");
  LinearSigmoidLayer l1(2, 2);
  LinearSigmoidLayer l2(2, 1);
  l2.set_is_last(true);
  layers = {&l1, &l2};

  Matrix input = training_data[4];
  Matrix truth = training_labels[4];
  truth.print();
  Matrix ff1 = l1.feed_forward(input);
  // ff1.print();
  Matrix ff2 = l2.feed_forward(ff1);
  ff2.print();

  for (auto i = 0; i < 500; i++) {
    Matrix prev = input;
    for (auto layer : layers) {
      prev = layer->feed_forward(prev);
    }

    outfile << i;
    for (auto j = 0; j < 4; j++) {
      outfile << "," << prev(0, j);
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
    outfile << "," << sqrt(sum);
    outfile << "\n";
    // std::cout << "err " << sqrt(sum) << std::endl;
  }

  ff1 = l1.feed_forward(input);
  // ff1.print();
  ff2 = l2.feed_forward(ff1);
  truth.print();
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