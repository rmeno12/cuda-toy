#include "mnist.hpp"

#include <fstream>
#include <iostream>

Mnist::Mnist(std::string data_folder) {
  training_images = read_images(data_folder + "/training_images");
  training_labels = read_labels(data_folder + "/training_labels");
  test_images = read_images(data_folder + "/test_images");
  test_labels = read_labels(data_folder + "/test_labels");
}

std::vector<char> Mnist::read_bytes(std::string filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  std::ifstream::pos_type pos = file.tellg();

  std::vector<char> bytes(pos);
  file.seekg(0, std::ios::beg);
  file.read(bytes.data(), pos);
  file.close();

  return bytes;
}

std::vector<Matrix> Mnist::read_images(std::string filename) {
  std::vector<char> bytes = read_bytes(filename);

  std::vector<Matrix> images;
  for (auto i = 16; i < bytes.size() - 784; i += 784) {
    Matrix image(784, 1);
    for (auto j = 0; j < 784; j++) {
      image(j, 0) = 0;
    }
    images.push_back(image);
  }

  return images;
}

std::vector<Matrix> Mnist::read_labels(std::string filename) {
  std::vector<char> bytes = read_bytes(filename);

  std::vector<Matrix> labels;
  for (auto i = 8; i < bytes.size(); i++) {
    Matrix label(10, 1);
    for (auto j = 0; j < 10; j++) {
      if (j == bytes[i]) {
        label(j, 0) = 1;
      } else {
        label(j, 0) = 0;
      }
    }
    labels.push_back(label);
  }

  return labels;
}

std::vector<Matrix> Mnist::get_training_images() { return training_images; }

std::vector<Matrix> Mnist::get_training_labels() { return training_labels; }

std::vector<Matrix> Mnist::get_test_images() { return test_images; }

std::vector<Matrix> Mnist::get_test_labels() { return test_labels; }
