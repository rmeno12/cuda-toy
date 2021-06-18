#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>

#include "linear_layer.hpp"

class Model {
 private:
  std::vector<LinearLayer*> layers;
  float learning_rate;

  static Matrix make_one_hot(Matrix input);
  static Matrix matrix_log(Matrix input);
  void backprop(Matrix preds, Matrix truths);
  void update_params();

 public:
  Model(float);

  Matrix predict(Matrix input);
  float rmse(Matrix input, Matrix truths);
  void train(Matrix input, Matrix truths);
  void train_batch(Matrix input, Matrix truths);
};

#endif