#ifndef LINEAR_SIGMOID_LAYER_HPP
#define LINEAR_SIGMOID_LAYER_HPP

#include "linear_layer.hpp"

class LinearSigmoidLayer : public LinearLayer {
 private:
  void initialize_weights() override;
  Matrix activate(Matrix input) override;
  Matrix d_sigmoid(Matrix input);

 public:
  LinearSigmoidLayer(size_t input_size, size_t size);

  Matrix backprop(Matrix m1, Matrix m2) override;
};

#endif