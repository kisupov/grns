/*
 *  Test for checking the correctness of parallel reduction
 */

#include "../src/pairwise.cuh"
#include "logger.cuh"

__global__ void test_cuda_block_reduce(double * in, int n){
    double result = cuda::block_reduce_sum_ru(in[threadIdx.x], RNS_MODULI_SIZE);
    printf("\n Thread = %i, Sum = %lf", threadIdx.x, result);
}

int main() {
    Logger::beginTestDescription(Logger::TEST_VERIFY_REDUCTION);
    double data[RNS_MODULI_SIZE];
    for(double & i : data){
        i = 1;
    }
    double * dx;
    cudaMalloc(&dx, sizeof(double) * RNS_MODULI_SIZE);
    cudaMemcpy(dx, data, sizeof(double) * RNS_MODULI_SIZE, cudaMemcpyHostToDevice);
    printf("\n Size of data = %i", RNS_MODULI_SIZE);
    test_cuda_block_reduce<<<1, RNS_MODULI_SIZE>>>(dx, RNS_MODULI_SIZE);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    Logger::printSpace();
    Logger::endTestDescription();
    return 0;
}