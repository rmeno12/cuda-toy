#ifndef LINEAR_SOFTMAX_LAYER_HPP
#define LINEAR_SOFTMAX_LAYER_HPP

#include "linear_layer.hpp"

class LinearSoftmaxLayer : public LinearLayer {
 private:
  void initialize_weights() override;
  Matrix activate(Matrix input) override;

 public:
  LinearSoftmaxLayer(size_t input_size, size_t size);

  Matrix backprop(Matrix activations, Matrix truths) override;
};

#endif