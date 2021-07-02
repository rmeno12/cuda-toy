#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>

#include "linear_layer.hpp"

class MnistModel {
 protected:
  std::vector<LinearLayer*> layers;
  float learning_rate;

 public:
  MnistModel();

  const Matrix predict(const Matrix& x);
  const Matrix train(const Matrix& x, const Matrix& y);
};

#endif