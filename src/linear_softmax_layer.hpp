#ifndef LINEAR_SOFTMAX_LAYER_HPP
#define LINEAR_SOFTMAX_LAYER_HPP

#include "linear_layer.hpp"

class LinearSoftmaxLayer : public LinearLayer {
 private:
  void initialize_weights() override;
  Matrix activate(Matrix) override;

 public:
  LinearSoftmaxLayer(size_t, size_t);

  Matrix backprop(Matrix, Matrix) override;
};

#endif