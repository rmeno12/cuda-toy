#ifndef LINEAR_SIGMOID_LAYER_HPP
#define LINEAR_SIGMOID_LAYER_HPP

#include "linear_layer.hpp"

class LinearSigmoidLayer : public LinearLayer {
 private:
  void initialize_weights() override;
  Matrix activate(Matrix) override;
  Matrix d_sigmoid(Matrix);

 public:
  LinearSigmoidLayer(size_t, size_t);

  Matrix backprop(Matrix, Matrix) override;
};

#endif