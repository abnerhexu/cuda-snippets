#include <stdio.h>
#include <cuda.h>

__global__ void vecAdd(float *a, float *b, float *c, int n) {
    // 考虑到将n个元素分配到n个线程中，每一个线程完成一对元素的计算
    // 因此，在创建线程块的时候创建一维的线程块即可
    // 假设我们创建的grid为(x1, 1, 1)，block为(x2, 1, 1)
    // 其中x1、x2为常量，x1 * x2 >= n
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void cpuVecAdd(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void assert_close(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("Error happens at %d, left %f != right %f\n", i, a[i], b[i]);
            exit(0);
        }
    }
}

int main() {
    int n = 16384;
    // cpu tester
    float *a, *b, *c;
    a = (float*)malloc(sizeof(float) * n);
    b = (float*)malloc(sizeof(float) * n);
    c = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }
    cpuVecAdd(a, b, c, n);

    // gpu tester
    float *d_a, *d_b, *d_c;
    float* gres_c;
    gres_c = (float*)malloc(sizeof(float) * n);
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(n/32, 1, 1);
    dim3 block(32, 1, 1);
    vecAdd<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaMemcpy(gres_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // result check
    assert_close(gres_c, c, n);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    free(gres_c);
    printf("Sccess!\n");
    return 0;
}