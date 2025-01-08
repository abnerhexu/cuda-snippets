#include <iostream>
#include <cuda.h>
#include "../utils/timer.h"
#include "../utils/data.h"
#include <assert.h>

// blockDim.x = 16, blockDim.y = 16
template<int BM, int BN, int BK, int TM, int TN> // TM = TN = 8, BK = 8; BM = blockDim.x*TM = 128, BN = blockDim.y*TN = 128
__global__ void tiled_mma(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tiled_a[BM * BK]; // 128 * 8 
    __shared__ float tiled_b[BN * BK]; // 8 * 128, but transposed
    float tmp[TM * TN] = {0.0f}; // finally, compute a 4 * 4 block **every thread**
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // threaed id in a block, assert it >= 128*8
    int tiled_ax = tid % 2;
    int tiled_ay = tid / 2;
    int tiled_bx = tid % 32;
    int tiled_by = tid / 32;
    int tiled_aidx = tiled_ay * BK + tiled_ax * 4;
    int tiled_bidx = tiled_bx * BK * 4 + tiled_by;
    int ay = TM * blockIdx.y * blockDim.y;
    int bx = TN * blockIdx.x * blockDim.x;
    float pb[4];
    for (int it = 0; it < K; it += BK) {
        // tiled_a[tiled_aidx] = a[(ay + tiled_ay) * K + it + tiled_ax];
        // tiled_b[tiled_bidx] = b[(tiled_by + it) * N + bx + tiled_bx];
        (float4 &)tiled_a[tiled_aidx] = (float4 &)a[(ay + tiled_ay) * K + it + 4 * tiled_ax];
        (float4 &)pb[0] = (float4 &)b[(tiled_by + it) * N + bx + 4 * tiled_bx];
        tiled_b[tiled_bidx         ] = pb[0];
        tiled_b[tiled_bidx + BK    ] = pb[1];
        tiled_b[tiled_bidx + 2 * BK] = pb[2];
        tiled_b[tiled_bidx + 3 * BK] = pb[3];
        // (float4 &)tiled_b[tiled_bidx] = (float4 &)b[(tiled_by + it) * N + bx + 4 * tiled_bx];
        __syncthreads();
        for (int p = 0; p < TM; p++) {
            for (int q = 0; q < TN; q++) {
                for (int k = 0; k < BK; k++) {
                    tmp[p * TN + q] += tiled_a[(p + threadIdx.y * TM) * BK + k] * tiled_b[(q + threadIdx.x * TN) * BK + k];
                }
            }
        }
        __syncthreads();
    }
    for (int p = 0; p < TM; p++) {
        for (int q = 0; q < TN; q++) {
            c[(ay + threadIdx.y * TM + p) * N + (bx + threadIdx.x * TN + q)] += tmp[p * TN + q]; // write back to global memory
        }
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
    dim3 blocksPerGrid(8, 8, 1);
    dim3 threadsPerBlock(16, 16, 1);
    CUDAProgTimer timer;
    timer.start();
    for (int i = 0; i < 1; i++)
    tiled_mma<128, 128, 8, 8, 8><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);
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
    // for (int p = 0; p < 1024; p++) {
    //     for (int q = 0; q < 1024; q++) {
    //         // std::cout << cc[p * 1024 + q] << " ";
    //         assert(cc[p * 1024 + q] == 1);
    //     }
    //     // std::cout << std::endl;
    // }
    
    return 0;
}