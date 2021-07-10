#include "matrix.hpp"

int main() {
  Matrix A({1, 2, 3, 4}, 2, 2);
  Matrix B({5, 6, 7, 8}, 2, 2);

  A.print();
  B.print();
  A.augment(B, 0);
  A.print();
}