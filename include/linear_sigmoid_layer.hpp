#ifndef LINEAR_SIGMOID_LAYER_HPP
#define LINEAR_SIGMOID_LAYER_HPP

#include "linear_layer.hpp"

class LinearSigmoidLayer : public LinearLayer {
 protected:
  void init_weights() override;
  void init_biases() override;

  const Matrix activate(const Matrix& input) const override;
  const Matrix d_activate(const Matrix& input) const override;

 public:
  LinearSigmoidLayer(const size_t& input_size, const size_t& output_size);

  const Matrix backward(const Matrix& m1, const Matrix& m2) override;
};

#endif