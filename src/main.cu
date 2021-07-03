#include "matrix.hpp"

int main() {
  Matrix A({1, 2, 3, 4}, 2, 2);
  Matrix B({2, 3, 3, 2}, 2, 2);
  Matrix C = A * B;

  A.print();
  B.print();
  C.print();
}