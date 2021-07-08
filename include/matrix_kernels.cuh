__global__ void matmul_k(float* lhs, float* rhs, float* res, size_t rows,
                         size_t mid, size_t cols);
__global__ void matadd_k(float* lhs, float* rhs, size_t rows, size_t cols,
                         bool sub, bool broadcast);
__global__ void matprod_k(float* lhs, float* rhs, size_t rows, size_t cols);
__global__ void matdiv_k(float* lhs, float* rhs, size_t rows, size_t cols);

__global__ void mataddscal_k(float* lhs, float rhs, size_t rows, size_t cols,
                             bool sub);
__global__ void matprodscal_k(float* lhs, float rhs, size_t rows, size_t cols);
__global__ void matdivscal_k(float* lhs, float rhs, size_t rows, size_t cols);