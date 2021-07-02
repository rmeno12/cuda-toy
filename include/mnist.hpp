#ifndef MNIST_HPP
#define MNIST_HPP

#include <string>
#include <tuple>
#include <vector>

#include "matrix.hpp"

class Mnist {
 private:
  std::vector<Matrix> training_images;
  std::vector<Matrix> training_labels;
  std::vector<Matrix> test_images;
  std::vector<Matrix> test_labels;

  std::vector<char> read_bytes(std::string filename);
  std::vector<Matrix> read_images(std::string filename);
  std::vector<Matrix> read_labels(std::string filename);

 public:
  Mnist(std::string);

  const std::tuple<Matrix, Matrix> get_training_batch(
      size_t batch_size = 64) const;

  const std::vector<Matrix> get_training_images() const;
  const std::vector<Matrix> get_training_labels() const;
  const std::vector<Matrix> get_test_images() const;
  const std::vector<Matrix> get_test_labels() const;
};

#endif