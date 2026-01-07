#include <iostream>
#include <omp.h>

int main() {
    // 设置线程数量（可选）
    // omp_set_num_threads(4);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", thread_id, total_threads);
    }

    return 0;
}