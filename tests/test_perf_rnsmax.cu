/*
 *  Test for measure the performance of finding the maximum element in an array of RNS numbers
 */

#include <stdio.h>
#include <iostream>
#include "../src/dpp/rnsmax.cuh"
#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"

#define ARRAY_SIZE 1000000

/*
 *  Naive calculation of the maximum element of an RNS number using interval evaluations.
 *  The interval evaluation is computed each time a comparison is made.
 */

__global__ void rns_max_naive_kernel(int *out, int *in, unsigned int N){
    auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    auto sharedIdx = threadIdx.x * RNS_MODULI_SIZE;
    extern __shared__ int shared_rns[];
    memset(&shared_rns[sharedIdx], 0, RNS_MODULI_SIZE * sizeof(int));
    while (numberIdx < N){
        if(cuda::rns_cmp(&in[numberIdx * RNS_MODULI_SIZE], &shared_rns[sharedIdx]) == 1){
            memcpy(&shared_rns[sharedIdx], &in[numberIdx * RNS_MODULI_SIZE], RNS_MODULI_SIZE * sizeof(int));
        }
        numberIdx +=  gridDim.x * blockDim.x;
    }
    __syncthreads();
    auto i = cuda::NEXT_POW2(blockDim.x) >> 1;
    while(i >= 1) {
        if ((threadIdx.x < i) && (threadIdx.x + i < blockDim.x) && cuda::rns_cmp(&shared_rns[(threadIdx.x + i) * RNS_MODULI_SIZE], &shared_rns[sharedIdx]) == 1) {
            memcpy(&shared_rns[sharedIdx], &shared_rns[(threadIdx.x + i) * RNS_MODULI_SIZE], RNS_MODULI_SIZE * sizeof(int));
        }
        i = i >> 1;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        memcpy(&out[blockIdx.x * RNS_MODULI_SIZE], &shared_rns[sharedIdx], RNS_MODULI_SIZE * sizeof(int));
    }
}

template <int gridSize, int blockSize>
void rns_max_naive(int *out, int *in, unsigned int N, int * buffer) {
    size_t memSize = blockSize * sizeof(int) * RNS_MODULI_SIZE;
    rns_max_naive_kernel <<< gridSize, blockSize, memSize >>> (buffer, in, N);
    rns_max_naive_kernel <<< 1, blockSize, memSize >>> (out, buffer, gridSize);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

/*
 *  Naive calculation of the maximum element of an RNS number using mixed-radix conversion.
 *  The mixed-radix representation is computed each time a comparison is made.
 */

__global__ void rns_max_mrc_naive_kernel(int *out, int *in, unsigned int N){
    auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    auto sharedIdx = threadIdx.x * RNS_MODULI_SIZE;
    extern __shared__ int sdatamrc[];
    memset(&sdatamrc[sharedIdx], 0, RNS_MODULI_SIZE * sizeof(int));
    while (numberIdx < N){
        if(cuda::mrc_compare_rns(&in[numberIdx * RNS_MODULI_SIZE], &sdatamrc[sharedIdx]) == 1){
            memcpy(&sdatamrc[sharedIdx], &in[numberIdx * RNS_MODULI_SIZE], RNS_MODULI_SIZE * sizeof(int));
        }
        numberIdx +=  gridDim.x * blockDim.x;
    }
    __syncthreads();
    auto i = cuda::NEXT_POW2(blockDim.x) >> 1;
    while(i >= 1) {
        if ((threadIdx.x < i) && (threadIdx.x + i < blockDim.x) && cuda::mrc_compare_rns(&sdatamrc[(threadIdx.x + i) * RNS_MODULI_SIZE], &sdatamrc[sharedIdx]) == 1) {
            memcpy(&sdatamrc[sharedIdx], &sdatamrc[(threadIdx.x + i) * RNS_MODULI_SIZE], RNS_MODULI_SIZE * sizeof(int));
        }
        i = i >> 1;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        memcpy(&out[blockIdx.x * RNS_MODULI_SIZE], &sdatamrc[sharedIdx], RNS_MODULI_SIZE * sizeof(int));
    }
}

template <int gridSize, int blockSize>
void rns_max_mrc_naive(int *out, int *in, unsigned int N, int * buffer) {
    size_t memSize = blockSize * sizeof(int) * RNS_MODULI_SIZE;
    rns_max_mrc_naive_kernel <<< gridSize, blockSize, memSize >>> (buffer, in, N);
    rns_max_mrc_naive_kernel <<< 1, blockSize, memSize >>> (out, buffer, gridSize);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

/*
 *  Calculation of the maximum element of an array of RNS numbers using mixed-radix conversion as described in rnsmax.cuh
 */

typedef struct {
    int digits[RNS_MODULI_SIZE];
    int idx;
} mrd_t;

DEVICE_CUDA_FORCEINLINE int rns_max_cmp_mrc(mrd_t *mrx, mrd_t *mry) {
    if(mry->idx < 0){
        return 1;
    }
    if(mrx->idx < 0){
        return -1;
    }
    return cuda::mrs_cmp(mrx->digits, mry->digits);
}

__global__ void rns_max_mrc_compute_kernel(mrd_t *out, int *in, unsigned int N){
    auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    while (numberIdx < N){
        cuda::perform_mrc(out[numberIdx].digits, &in[numberIdx * RNS_MODULI_SIZE]);
        out[numberIdx].idx = numberIdx;
        numberIdx +=  gridDim.x * blockDim.x;
    }
}

__global__ void rns_max_mrc_tree_kernel(mrd_t *out, mrd_t *in, unsigned int N){
    auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ mrd_t shared_mrc[];
    shared_mrc[threadIdx.x].idx = -1;
    while (numberIdx < N){
        if(rns_max_cmp_mrc(&in[numberIdx], &shared_mrc[threadIdx.x]) == 1){
            shared_mrc[threadIdx.x] = in[numberIdx];
        }
        numberIdx +=  gridDim.x * blockDim.x;
    }
    __syncthreads();
    auto i = cuda::NEXT_POW2(blockDim.x) >> 1;
    while(i >= 1) {
        if ((threadIdx.x < i) && (threadIdx.x + i < blockDim.x) && rns_max_cmp_mrc(&shared_mrc[threadIdx.x + i], &shared_mrc[threadIdx.x]) == 1) {
            shared_mrc[threadIdx.x] = shared_mrc[threadIdx.x + i];
        }
        i = i >> 1;
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = shared_mrc[threadIdx.x];
}

__global__ void rns_max_mrc_set_kernel(int *out, int *in, mrd_t *mrd){
    int idx = mrd[0].idx * RNS_MODULI_SIZE + threadIdx.x;
    out[threadIdx.x] = in[idx];
}

template <int gridSize1, int blockSize1, int gridSize2, int blockSize2>
void rns_max_mrc(int *out, int *in, unsigned int N, mrd_t *buffer) {
    size_t memSize = blockSize2 * sizeof(mrd_t);
    rns_max_mrc_compute_kernel <<< gridSize1, blockSize1 >>> ( buffer, in, N);
    rns_max_mrc_tree_kernel <<< gridSize2, blockSize2, memSize >>> (buffer, buffer, N);
    rns_max_mrc_tree_kernel <<< 1, blockSize2, memSize >>> (buffer, buffer, gridSize2);
    rns_max_mrc_set_kernel <<< 1, RNS_MODULI_SIZE >>> (out, in, buffer);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

/*
 * Benchmarks
 */

static void printResult(int * result){
    mpz_t binary;
    mpz_init(binary);
    rns_to_binary(binary, result);
    printf("result: %s", mpz_get_str(NULL, 10, binary));
    mpz_clear(binary);
}

//Test of the interval evaluation implementation
void test_rns_max(int * drx, int array_size) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[CUDA] rns_max");

    //Execution config
    const int gridSize1 = 4096;
    const int blockSize1 = 64;
    const int gridSize2 = 1024;
    const int blockSize2 = 64;
    printf("(exec. config: gridSize1 = %i, blockSize1 = %i, gridSize2 = %i, blockSize2 = %i)\n", gridSize1, blockSize1, gridSize2, blockSize2);

    //Device data
    int * dresult;
    xinterval_t * dbuf;
    cudaMalloc(&dresult, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&dbuf, sizeof(xinterval_t) * array_size);
    printf("memory buffer size (MB): %lf\n", double(sizeof(xinterval_t)) * array_size /  double(1024 * 1024));
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    cuda::rns_max<gridSize1, blockSize1, gridSize2, blockSize2>(dresult, drx, array_size, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host and verify
    int hresult[RNS_MODULI_SIZE];
    cudaMemcpy(hresult, dresult, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    printResult(hresult);
    Logger::printSpace();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Cleanup
    cudaFree(dresult);
    cudaFree(dbuf);
}

//Test of the naive interval implementation
void test_rns_max_naive(int * drx, int array_size) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[CUDA] rns_max_naive");

    //Execution config
    const int gridSize = 256;
    const int blockSize = 32;
    printf("(exec. config: gridSize = %i, blockSize = %i)\n", gridSize, blockSize);

    //Device data
    int * dresult;
    int * dbuf;
    cudaMalloc(&dresult, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&dbuf, sizeof(int) * gridSize);
    printf("memory buffer size (MB): %lf\n", double(sizeof(int)) * gridSize /  double(1024 * 1024));
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    rns_max_naive<gridSize, blockSize>(dresult, drx, array_size, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host and verify
    int hresult[RNS_MODULI_SIZE];
    cudaMemcpy(hresult, dresult, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    printResult(hresult);
    Logger::printSpace();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Cleanup
    cudaFree(dresult);
    cudaFree(dbuf);
}

//Test of the naive mrc implementation
void test_rns_max_mrc_naive(int * drx, int array_size) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[CUDA] rns_max_mrc_naive");

    //Execution config
    const int gridSize = 256;
    const int blockSize = 32;
    printf("(exec. config: gridSize = %i, blockSize = %i)\n", gridSize, blockSize);

    //Device data
    int * dresult;
    int * dbuf;
    cudaMalloc(&dresult, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&dbuf, sizeof(int) * gridSize);
    printf("memory buffer size (MB): %lf\n", double(sizeof(int)) * gridSize /  double(1024 * 1024));
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    rns_max_mrc_naive<gridSize, blockSize>(dresult, drx, array_size, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host and verify
    int hresult[RNS_MODULI_SIZE];
    cudaMemcpy(hresult, dresult, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    printResult(hresult);
    Logger::printSpace();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Cleanup
    cudaFree(dresult);
    cudaFree(dbuf);
}

//Test of the MRC implementation
void test_rns_max_mrc(int * drx, int array_size) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[CUDA] rns_max_mrc");

    //Execution config
    const int gridSize1 = 128;
    const int blockSize1 = 64;
    const int gridSize2 = 256;
    const int blockSize2 = 128; //64 - for 128 moduli, 32 for 256 moduli, 128 - for other
    printf("(exec. config: gridSize1 = %i, blockSize1 = %i, gridSize2 = %i, blockSize2 = %i)\n", gridSize1, blockSize1, gridSize2, blockSize2);

    //Device data
    int * dresult;
    mrd_t * dbuf;
    cudaMalloc(&dresult, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&dbuf, sizeof(mrd_t) * array_size);
    printf("memory buffer size (MB): %lf\n", double(sizeof(mrd_t)) * array_size /  double(1024 * 1024));
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    rns_max_mrc<gridSize1, blockSize1, gridSize2, blockSize2>(dresult, drx, array_size, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host and verify
    int hresult[RNS_MODULI_SIZE];
    cudaMemcpy(hresult, dresult, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    printResult(hresult);
    Logger::printSpace();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Cleanup
    cudaFree(dresult);
    cudaFree(dbuf);
}

//Main benchmark
void run_test(size_t array_size){
    //Host and device data
    auto * hx = new mpz_t[array_size];
    int * hrx = new int[array_size * RNS_MODULI_SIZE];
    int * drx;
    size_t memsize = sizeof(int) * array_size * RNS_MODULI_SIZE;
    cudaMalloc(&drx, memsize);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Memory allocation
    for(int i = 0; i < array_size; i++){
        mpz_init(hx[i]);
    }
    //Generate inputs
    fill_random_array(hrx, array_size);
    for(auto i = 0; i < ARRAY_SIZE; i++){
        for(auto j = 0; j < RNS_MODULI_SIZE; j++){
            hrx[i * RNS_MODULI_SIZE + j] = hrx[i * RNS_MODULI_SIZE + j] % RNS_MODULI[j];
        }
        rns_to_binary(hx[i], &hrx[i * RNS_MODULI_SIZE]);
    }
    // Copying to the GPU
    cudaMemcpy(drx, hrx, memsize, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //---------------------------------------------------------
    //Compute exact result
    //---------------------------------------------------------
    Logger::printDash();
    mpz_t hmax;
    mpz_init(hmax);
    mpz_set_ui(hmax, 0);
    for(int i = 0; i < array_size; i++){
        if(mpz_cmp(hx[i], hmax) > 0){
            mpz_set(hmax, hx[i]);
        }
    }
    printf("[GNU MP] max: \nresult: %s", mpz_get_str(NULL, 10, hmax));
    Logger::printSpace();

    //---------------------------------------------------------
    //Run CUDA tests
    //---------------------------------------------------------
    test_rns_max(drx, array_size);
    test_rns_max_naive(drx, array_size);
    test_rns_max_mrc_naive(drx, array_size);
    test_rns_max_mrc(drx, array_size);
    Logger::printSpace();
    //---------------------------------------------------------

    //Cleanup
    for(int i = 0; i < array_size; i++){
        mpz_clear(hx[i]);
    }
    mpz_clear(hmax);
    delete [] hx;
    delete [] hrx;
    cudaFree(drx);
}

int main() {
    cudaDeviceReset();
    rns_const_init();
    Logger::beginTestDescription(Logger::TEST_PERF_RNSMAX);
    Logger::printParam("ARRAY_SIZE", ARRAY_SIZE);
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("RNS_MODULI_PRODUCT_LOG2", RNS_MODULI_PRODUCT_LOG2);
    Logger::endSection(true);
    run_test(ARRAY_SIZE);
    Logger::endTestDescription();
    return 0;
}