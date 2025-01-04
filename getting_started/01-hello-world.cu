#include <stdio.h>
#include <cuda.h>

__global__ void hello_world() {
    // 行主序计算当前线程在所有线程中是第几个线程
    int x = threadIdx.x + threadIdx.y * blockDim.x;
    printf("Hello, world from %d\n", x);
}

int main() {
    dim3 grid(1, 1, 1);
    dim3 block(4, 2, 1);

    hello_world<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}