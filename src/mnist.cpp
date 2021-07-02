#include "mnist.hpp"

#include <fstream>
#include <iostream>
#include <random>

Mnist::Mnist(std::string data_folder) {
  read_training(data_folder);
  // training_images = read_images(data_folder + "/training_images");
  // training_labels = read_labels(data_folder + "/training_labels");
  // test_images = read_images(data_folder + "/test_images");
  // test_labels = read_labels(data_folder + "/test_labels");
}

void Mnist::read_training(std::string filename) {
  std::ifstream img_file;
  img_file.open(filename + "/training_images", std::ios::in | std::ios::binary);
  img_file.seekg(16, std::ios::cur);

  int numbytes = NUM_TRAINING * 784 + 16;
  int img_num = 0;
  int i = 16;
  while (i < numbytes) {
    training_images[img_num] = new Matrix(784, 1);
    unsigned char* img = new unsigned char[784];
    img_file.read((char*)img, 784);
    for (auto j = 0; j < 784; j++) {
      (*training_images[img_num])(j, 0) = img[j];
    }
    i += 784;
    img_num++;
  }

  std::ifstream lbl_file;
  lbl_file.open(filename + "/training_labels", std::ios::in | std::ios::binary);
  lbl_file.seekg(8, std::ios::cur);

  numbytes = NUM_TRAINING + 8;
  img_num = 0;
  i = 8;
  while (i < numbytes) {
    training_labels[img_num] = new Matrix(10, 1);
    unsigned char val;
    lbl_file.read((char*)&val, 1);

    (*training_labels[img_num])(val, 0) = 1;
    i++;
    img_num++;
  }
}

const std::tuple<Matrix, Matrix> Mnist::get_training_batch(
    size_t batch_size) const {
  if (batch_size < 1)
    throw std::invalid_argument("Batch size must be at least 1");

  std::default_random_engine eng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, NUM_TRAINING - 1);

  int rand = dist(eng);
  Matrix images = *training_images[rand];
  Matrix labels = *training_labels[rand];
  for (auto i = 1; i < batch_size; i++) {
    rand = dist(eng);
    images.augment(*training_images[rand], 1);
    labels.augment(*training_labels[rand], 1);
  }

  return std::make_tuple(images, labels);
}

void Mnist::printImage(const Matrix& img) {
  const std::vector<char> scale = {' ', '.', ':', '-', '=',
                                   '+', '*', '#', '%', '@'};
  std::string out = "";
  for (auto i = 0; i < img.get_rows(); i++) {
    float val = img(i, 0);
    int which = std::floor(val * 10.0 / 256.0);
    std::cout << scale[which];
    if ((i + 1) % 28 == 0) {
      std::cout << '\n';
    }
  }

  std::cout << std::endl;
}
