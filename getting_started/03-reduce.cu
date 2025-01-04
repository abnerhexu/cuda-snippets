#include <stdio.h>
#include <cuda.h>
#include <cub/block/block_reduce.cuh>
#include "../utils/data.h"

template<int BLOCKDIM_X>
__global__ void vecRed(float *a, float *b, float* res, int n) {
    // 假设我们拥有grid(1, 1, 1)和block(32, 1, 1)，将形状为(16384, 1)的两个向量相加，并求和
    // 每个线程需要处理16384/32个元素
    int idx = threadIdx.x;
    float thread_sum = 0.0f;
    int range = n / BLOCKDIM_X;
    __shared__ float partials[BLOCKDIM_X];
    for (int i = 0; i < range; i++) {
        if (idx * range + i < n) {
            thread_sum += a[idx * range + i] + b[idx * range + i];
        }
    }
    partials[idx] = thread_sum;
    // 使用__syncthreads()来同步线程
    __syncthreads();
    // 方案一：让第一个线程加完所有线程的结果
    if (idx == 0) {
        for (int i = 1; i < BLOCKDIM_X; i++) {
            thread_sum += partials[i];
        }
        res[0] = thread_sum;
    }
    // 方案二：使用cub模板
    // 使用cub模板时其实并不需要前面开辟的__shared__，可以直接对私有变量做reduce
    // typedef cub::BlockReduce<int, BLOCKDIM_X> BlockReduce;
    // 需要一个shared memory来保存中间结果（cub需要的）
    // __shared__ typename BlockReduce::TempStorage temp_storage;
    // int block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    // if (idx == 0) {
    //     res[0] = block_sum;
    // }
}

void cpuVecRed(float* a, float* b, float* res, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] + b[i];
    }
    res[0] = sum;
}

int main() {
    int n = 16384;
    // cpu vecRed
    auto a = generate_random_array<float>(n, 1, 3);
    auto b = generate_random_array<float>(n, 1, 3);
    auto cres = (float*)malloc(sizeof(float));
    cpuVecRed(a.data(), b.data(), cres, n);

    // gpu vecRed
    float *d_a, *d_b, *d_res;
    float *gres = (float*)malloc(sizeof(float));
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);
    vecRed<32><<<grid, block>>>(d_a, d_b, d_res, n);
    cudaMemcpy(gres, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // result check
    std::cout << "cres: " << cres[0] << ", gres: " << gres[0] << std::endl;
    assert_close<float>(cres, gres, 1, 0.5e-2);
    free(cres);
    free(gres);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    return 0;
}
