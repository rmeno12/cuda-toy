#ifndef LINEAR_RELU_LAYER_H
#define LINEAR_RELU_LAYER_H

#include "linear_layer.hpp"

class LinearReluLayer : public LinearLayer {
 private:
  void initialize_weights() override;
  Matrix activate(Matrix) override;
  static Matrix d_relu(Matrix);

 public:
  LinearReluLayer(size_t, size_t);

  Matrix backprop(Matrix, Matrix) override;
};

#endif