#ifndef LINEAR_RELU_LAYER_HPP
#define LINEAR_RELU_LAYER_HPP

#include "linear_layer.hpp"

class LinearReluLayer : public LinearLayer {
 protected:
  void init_weights() override;
  void init_biases() override;

  const Matrix activate(const Matrix& input) const override;
  const Matrix d_activate(const Matrix& input) const override;

 public:
  LinearReluLayer(const size_t& input_size, const size_t& output_size);

  const Matrix backward(const Matrix& m1, const Matrix& m2) override;
};

#endif