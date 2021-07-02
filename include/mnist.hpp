#ifndef MNIST_HPP
#define MNIST_HPP

#include <string>
#include <tuple>
#include <vector>

#include "matrix.hpp"

#define NUM_TRAINING 600
#define NUM_TEST 100

class Mnist {
 private:
  Matrix* training_images[NUM_TRAINING];
  Matrix* training_labels[NUM_TRAINING];
  Matrix* test_images[NUM_TEST];
  Matrix* test_labels[NUM_TEST];

  std::vector<char> read_bytes(std::string filename);
  void read_training(std::string filename);

 public:
  Mnist(std::string);

  const std::tuple<Matrix, Matrix> get_training_batch(
      size_t batch_size = 64) const;

  static void printImage(const Matrix& img);
};

#endif