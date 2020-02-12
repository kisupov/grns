/*
 *  Run time measurement macros for CPU (with OpenMP support) and CUDA
 */

#ifndef GRNS_TEST_TIMERS_CUH
#define GRNS_TEST_TIMERS_CUH

#include <iostream>
#include <chrono>
#include <time.h>

/*
 * Return CPU time in milliseconds between start and end
 */
inline double calcTimeCPU(struct timespec start, struct timespec end){
    long long start_nanos = start.tv_sec * 1000000000LL + start.tv_nsec;
    long long end_nanos = end.tv_sec * 1000000000LL + end.tv_nsec;
    return (end_nanos - start_nanos) * 1e-6f;
}

/*
 * Return GPU time in milliseconds between start and stop
 */
float calcTimeCUDA(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
};

#define PrintTimerName(msg) std::cout << msg << " testing..." << std::endl;

#define InitCpuTimer()       struct timespec start, end; double _cpu_time = 0;

#define StartCpuTimer();      clock_gettime(CLOCK_MONOTONIC, &start);

#define EndCpuTimer();        clock_gettime(CLOCK_MONOTONIC, &end);_cpu_time += calcTimeCPU(start, end);

#define PrintCpuTimer(msg) std::cout << msg << "(ms): " << _cpu_time << std::endl;_cpu_time=0;

#define InitCudaTimer();                         \
      float _cuda_time = 0;                         \
      cudaEvent_t cuda_timer_start, cuda_timer_end; \
      cudaEventCreate(&cuda_timer_start);           \
      cudaEventCreate(&cuda_timer_end);

#define StartCudaTimer(); cudaEventRecord(cuda_timer_start, 0);

#define EndCudaTimer();                                           \
      checkDeviceHasErrors(cudaEventRecord(cuda_timer_end, 0));      \
      checkDeviceHasErrors(cudaEventSynchronize(cuda_timer_end));  \
      _cuda_time += calcTimeCUDA(cuda_timer_start, cuda_timer_end);

#define PrintCudaTimer(msg);   \
    std::cout << msg << "(ms): " << _cuda_time << std::endl; \
    _cuda_time = 0;

/*
 * Delay in seconds
 */
void waitFor (unsigned int secs) {
    unsigned int retTime = time(0) + secs;
    while (time(0) < retTime);
}

#endif //GRNS_TEST_TIMERS_CUH
