#include <iostream>
#include <cuda.h>
#include "../utils/data.h"

__global__ void simple_loop_mm(float* a, float* b, float* c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

void standard_model_mm(float* a, float* b, float* c, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

int main() {
    int M = 1024;
    int K = 1024;
    int N = 1024;

    auto a = generate_random_array<float>(M * K, -1.0, 1.0);
    auto b = generate_random_array<float>(K * N, -1.0, 1.0);
    auto c = generate_random_array<float>(M * N, -1.0, 1.0);

    // Call the standard model
    standard_model_mm(a.data(), b.data(), c.data(), M, K, N);

    // Initialize matrices a, b, and c
    std::vector<float> cc;
    cc.assign(c.begin(), c.end());
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, K * N * sizeof(float));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, cc.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the GPU model
    dim3 blockDim(4, 4, 1);
    dim3 gridDim(8, 8, 1);
    simple_loop_mm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);

    // Copy the result back to the host
    cudaMemcpy(cc.data(), d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Result check
    assert_close<float>(c.data(), cc.data(), M * N, 1e-5);
    return 0;
}