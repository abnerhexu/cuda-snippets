#include <cuda.h>
#include "../utils/data.h"
#include "../utils/timer.h"
template<int BLOCK_SIZE>
__global__ void shared_memory_mma(float *a, float *b, float *c, int M, int K, int N) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    int J = blockIdx.y * blockDim.y + threadIdx.y;
    int i = threadIdx.x;
    int j = threadIdx.y;

    __shared__ float partialA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float partialB[BLOCK_SIZE][BLOCK_SIZE];
    float psum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // load data into shared memory
        partialA[i][j] = (I < M && k + j < K) ? a[I * K + k + j] : 0.0f;
        partialB[i][j] = (k + i < K && J < N) ? b[(k + i) * N + J] : 0.0f;
        __syncthreads();
        for (int l = 0; l < BLOCK_SIZE; l++) {
            psum += partialA[i][l] * partialB[l][j];
        }
    }
    if (I < M && J < N) {
        c[I * N + J] += psum;
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
    dim3 blocksPerGrid(64, 64, 1);
    dim3 threadsPerBlock(16, 16, 1);
    CUDAProgTimer timer;
    timer.start();
    for (int i = 0; i < 1; i++)
    shared_memory_mma<16><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);
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