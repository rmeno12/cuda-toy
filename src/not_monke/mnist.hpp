#ifndef MNIST_HPP
#define MNIST_HPP

#include <string>
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

  std::vector<Matrix> get_training_images();
  std::vector<Matrix> get_training_labels();
  std::vector<Matrix> get_test_images();
  std::vector<Matrix> get_test_labels();
};

#endif