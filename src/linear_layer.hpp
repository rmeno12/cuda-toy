#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "matrix.hpp"

class LinearLayer {
 private:
  virtual void initialize_weights() {
    for (auto i = 0; i < input_size; i++) {
      for (auto j = 0; i < size; i++) {
        weights(i, j) = 0.0;
      }
    }
  }
  virtual void initialize_biases() {
    for (size_t i = 0; i < size; i++) {
      biases(i, 0) = 0.0;
    }
  }
  virtual Matrix activate(Matrix) = 0;

 protected:
  Matrix weights, biases;
  Matrix inp, z, error;
  size_t input_size, size;

 public:
  LinearLayer(size_t input_size, size_t size)
      : input_size(input_size),
        size(size),
        weights(input_size, size),
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
  virtual Matrix backprop(Matrix, Matrix) = 0;
  virtual void update_params(float learning_rate) {
    weights = weights - (error * inp.transpose()) * learning_rate;
    biases = biases - error * learning_rate;
  }
};

#endif