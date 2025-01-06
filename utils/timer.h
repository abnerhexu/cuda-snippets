#include <cuda.h>
#include <iostream>

class CUDAProgTimer {
public:
    cudaEvent_t t_start, t_end;
    float elapsed_time = 0.0f;
    CUDAProgTimer() {
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_end);
    };
    void start() {
        cudaEventRecord(t_start, 0);
    };
    void stop() {
        cudaEventRecord(t_end, 0);
        cudaEventSynchronize(t_end);
        cudaEventElapsedTime(&elapsed_time, t_start, t_end);
    };
    void info() {
        std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;
    }
};