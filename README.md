# cuda-toy

The goal of this project is to implement a simple neural network in C++ that is GPU accelerated via CUDA.

### Goals
- [DONE] Implement non-accelerated net
- [DONE] Optimize net architecture a little for MNIST data
- [DONE] Implement CUDA-accelerated matrix functions
- [DONE] Quantitatively compare accelerated vs non-accelerated performance

## Results
### CPU with `std::vector`
![CPU_vec_graph](./results/cpu_vec_500.png)
Mean iteration time ~ 3430 ms
### CPU with arrays
![CPU_graph](./results/cpu_500.png)
Mean iteration time ~ 428 ms
### CPU with GPU matrix multiplication
![GPU_matmul_graph](./results/gpu_matmul_500.png)
Mean iteration time ~ 18 ms
### CPU with GPU matrix arithmetic
![GPU_graph](./results/gpu_500.png)
Mean iteration time ~ 12 ms
