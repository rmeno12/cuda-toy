#include <chrono>
#include <fstream>
#include <iostream>

#include "linear_relu_layer.hpp"
#include "linear_sigmoid_layer.hpp"
#include "linear_softmax_layer.hpp"
#include "matrix.hpp"
#include "mnist.hpp"
#include "mnist_model.hpp"

float learning_rate = 0.03;

int x_size = 784;
int l1_size = 128;
int l2_size = 10;
LinearReluLayer l1(x_size, l1_size);
LinearSoftmaxLayer l2(l1_size, l2_size);

Matrix forward_backward(Matrix x, Matrix y) {
  // forward pass
  Matrix a1 = l1.forward(x);
  Matrix a2 = l2.forward(a1);

  Matrix loss = -(y.product(Matrix::log2(a2)).sum(0));

  // backward pass
  Matrix dl_dz2 = l2.backward(y, a2);
  Matrix dl_dz1 = l1.backward(l2.get_weights(), dl_dz2);

  return loss;
}

Matrix predict(Matrix x) {
  Matrix a1 = l1.forward(x);
  Matrix a2 = l2.forward(a1);

  return a2;
}

void manual_layers() {
  l2.set_last_layer(true);
  Mnist mnist("/workspaces/cuda-toy/data");

  for (auto i = 0; i < 10; i++) {
    auto [x, y] = mnist.get_training_batch();
    // Mnist::printImage(x);
    Matrix loss = forward_backward(x, y);
    loss.print();

    l1.update_params(learning_rate);
    l2.update_params(learning_rate);
  }

  auto [x, y] = mnist.get_training_batch(1);
  predict(x).print();
  y.print();
}

void output(std::vector<float> losses, std::vector<int> times) {
  std::ofstream outcsv;
  outcsv.open("output.csv");
  for (auto i = 0; i < losses.size(); i++) {
    outcsv << losses[i] << "," << times[i] << "\n";
  }
  outcsv.close();
}

int main() {
  // manual_layers();
  Mnist mnist("/workspaces/cuda-toy/data");
  std::cout << "read data" << std::endl;
  MnistModel model;
  std::vector<float> losses = {};
  std::vector<int> times = {};

  for (auto i = 0; i < 500; i++) {
    auto [batch, truths] = mnist.get_training_batch();
    auto starttime = std::chrono::high_resolution_clock::now();
    Matrix loss = model.train(batch, truths);
    auto endtime = std::chrono::high_resolution_clock::now();
    int batchtime = std::chrono::duration_cast<std::chrono::microseconds>(
                        endtime - starttime)
                        .count();
    float batch_loss = loss.mean(1)(0, 0);
    std::cout << "loss: " << batch_loss << std::endl;
    losses.push_back(batch_loss);
    times.push_back(batchtime);
  }

  auto [batch, truths] = mnist.get_training_batch(4);
  model.predict(batch).print();
  // batch.print();
  truths.print();
  output(losses, times);
}