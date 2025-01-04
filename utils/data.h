#include <random>
#include <type_traits>
#include <vector>
#include <iostream>

template<typename T>
std::vector<T> generate_random_array(size_t size, T min, T max) {
    static_assert(std::is_arithmetic<T>::value, "Type must be arithmetic (integral or floating-point)");

    // static std::random_device rd;  // 随机设备，用于生成种子
    static std::mt19937 gen(42); // 使用Mersenne Twister引擎

    std::vector<T> result(size);  // 用于存储随机数的数组

    if constexpr (std::is_integral<T>::value) {
        // 生成整数随机数
        std::uniform_int_distribution<T> dis(min, max);
        for (size_t i = 0; i < size; ++i) {
            result[i] = dis(gen);
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        // 生成浮点数随机数
        std::uniform_real_distribution<T> dis(min, max);
        for (size_t i = 0; i < size; ++i) {
            result[i] = dis(gen);
        }
    }

    return result;
}

template<typename T>
void assert_close(T *a, T *b, int n, T epsilon) {
    if constexpr (std::is_integral<T>::value) {
        for (int i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                std::cout << "Error happens at " << i << "; left: " << a[i] << ", right: " << b[i] << std::endl;
                exit(0);
            }
        }
    }
    else if constexpr (std::is_floating_point<T>::value) {
        for (int i = 0; i < n; i++) {
            if (abs(a[i] - b[i]) > 0.5e-1) {
                std::cout << "Error happens at " << i << "; left: " << a[i] << ", right: " << b[i] << std::endl;
                exit(0);
            }
        }
    }
    std::cout << "Success!" << std::endl;
}
template<typename T>
void assert_the_same(T *a, T *b, int n) {
    if constexpr (std::is_integral<T>::value) {
        for (int i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                std::cout << "Error happens at " << i << "; left: " << a[i] << ", right: " << b[i] << std::endl;
                exit(0);
            }
        }
    }
    else if constexpr (std::is_floating_point<T>::value) {
        for (int i = 0; i < n; i++) {
            if (abs(a[i] - b[i]) > 1e-6) {
                std::cout << "Error happens at " << i << "; left: " << a[i] << ", right: " << b[i] << std::endl;
                exit(0);
            }
        }
    }
    std::cout << "Success!" << std::endl;
}