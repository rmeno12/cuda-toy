#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>

#include "linear_layer.hpp"

class Model {
 private:
  std::vector<LinearLayer*> layers;
  float learning_rate;

  static Matrix make_one_hot(Matrix);
  static Matrix matrix_log(Matrix);
  void backprop(Matrix, Matrix);
  void update_params();
  void update_params_batch();

 public:
  Model(float);

  Matrix predict(Matrix);
  float rmse(Matrix, Matrix);
  void train(Matrix, Matrix);
  void train_batch(Matrix, Matrix);
};

#endif