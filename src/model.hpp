#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>

#include "linear_layer.hpp"

class Model {
 private:
  std::vector<LinearLayer*> layers;
  float learning_rate;

  static Matrix make_one_hot(Matrix);
  void backprop(Matrix, Matrix);
  void update_params();

 public:
  Model(float);

  Matrix predict(Matrix);
  void train(Matrix, Matrix);
  // void train_batch(Matrix, Matrix);
};

#endif