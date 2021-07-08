#include "matrix.hpp"

int main() {
  Matrix A({1, 2, 3, 4}, 2, 2);
  Matrix B({2, 3, 3, 2}, 2, 2);

  A.print();
  B.print();
  (A - B).print();
  (A - 1).print();
  (2 - A).print();
}