#ifndef LINEAR_RELU_LAYER_HPP
#define LINEAR_RELU_LAYER_HPP

#include "linear_layer.hpp"

class LinearReluLayer : public LinearLayer {
 private:
  void initialize_weights() override;
  Matrix activate(Matrix input) override;
  static Matrix d_relu(Matrix input);

 public:
  LinearReluLayer(size_t input_size, size_t size);

  Matrix backprop(Matrix next_weights, Matrix next_error) override;
};

#endif