#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include <math.h>

#include "matrix.hpp"

const float LOG2E = std::log2(std::exp(1));

class LinearLayer {
 private:
  virtual void initialize_weights() {
    for (auto i = 0; i < size; i++) {
      for (auto j = 0; j < input_size; j++) {
        weights(i, j) = 0.0;
      }
    }
  }
  virtual void initialize_biases() {
    for (size_t i = 0; i < size; i++) {
      biases(i, 0) = 0.0;
    }
  }
  virtual Matrix activate(Matrix input) = 0;

 protected:
  Matrix weights, biases;
  Matrix inp, z, error;
  size_t input_size, size;
  bool is_last_layer = false;

 public:
  LinearLayer(size_t input_size, size_t size)
      : input_size(input_size),
        size(size),
        weights(size, input_size),
        biases(size, 1),
        inp(input_size, 1),
        z(size, 1),
        error(size, 1) {
    initialize_weights();
    initialize_biases();
  }

  virtual Matrix feed_forward(Matrix input) {
    inp = input;
    z = weights * input + biases;
    return activate(z);
  }
  virtual Matrix backprop(Matrix m1, Matrix m2) = 0;
  virtual void update_params(float learning_rate) {
    Matrix weight_update = (error * inp.transpose()) * learning_rate;
    weights = weights - weight_update;
    biases = biases - error.collapse_horizontal_avg() * learning_rate;
  }

  Matrix get_weights() { return weights; }
  Matrix get_error() { return error; }
  bool get_is_last() { return is_last_layer; }
  void set_is_last(bool is_last) { is_last_layer = is_last; }
};

#endif