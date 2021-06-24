#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "matrix.hpp"

class LinearLayer {
 protected:
  size_t input_size, output_size;
  Matrix weights, biases;
  Matrix x, z, a;
  Matrix d_loss;
  bool last_layer;

  virtual void init_weights() = 0;
  virtual void init_biases() = 0;

  virtual const Matrix activate(const Matrix& input) const = 0;
  virtual const Matrix d_activate(const Matrix& input) const = 0;

  LinearLayer(const size_t& input_size, const size_t& output_size)
      : input_size(input_size),
        output_size(output_size),
        weights(output_size, input_size),
        biases(output_size, 1),
        x(input_size, 1),
        z(output_size, 1),
        a(output_size, 1),
        d_loss(output_size, 1),
        last_layer(false) {}

 public:
  virtual const Matrix forward(const Matrix& input) {
    x = input;
    z = weights * x + biases;
    a = activate(z);
    return a;
  };
  virtual const Matrix backward(const Matrix& input) = 0;
  virtual void update_params(const float& learning_rate) {
    weights -= learning_rate * (d_loss * x.transpose());
    biases -= learning_rate * d_loss.mean(1);
  }

  const Matrix get_weights() const { return weights; }
  const Matrix get_d_loss() const { return d_loss; }
  void set_last_layer(const bool& set) { last_layer = set; }
};

#endif