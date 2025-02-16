#include <iostream>
#include <cuda.h>
#include "../utils/data.h"
#include "../utils/timer.h"

__global__ void simple_loop_mma(float* a, float* b, float* c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        // 注意，在这里可能c矩阵同一个位置被写入多次，但是我们并不在乎，因为最终结果是一样的
        // 原因在于可能有多个元素都满足row < M并且col < N
        c[row * N + col] += sum;
    }
}

void standard_model_mma(float* a, float* b, float* c, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] += sum;
        }
    }
}

int main() {
    int M = 1024;
    int K = 1024;
    int N = 1024;

    auto a = generate_random_array<float>(M * K, 1.0, 1.0);
    auto b = generate_random_array<float>(K * N, 1.0, 1.0);
    auto c = generate_random_array<float>(M * N, 1.0, 1.0);

    std::vector<float> cc;
    std::copy(c.begin(), c.end(), std::back_inserter(cc));

    // Call the standard model
    standard_model_mma(a.data(), b.data(), c.data(), M, K, N);

    // Initialize matrices a, b, and c
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, K * N * sizeof(float));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, cc.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the GPU model
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(32, 32, 1);
    CUDAProgTimer timer;
    timer.start();
    for (int i = 0; i < 1; i++)
    simple_loop_mma<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
    timer.stop();
    timer.info();
    // Copy the result back to the host
    cudaMemcpy(cc.data(), d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free the device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Result check
    assert_close<float>(c.data(), cc.data(), M * N, 1e-5);
    return 0;
}