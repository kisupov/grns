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
#define RNS_MAX_NUM_BLOCKS_1 4096
#define RNS_MAX_BLOCK_SIZE_1 64
#define RNS_MAX_NUM_BLOCKS_2 1024
#define RNS_MAX_BLOCK_SIZE_2 64

/*
 *  Calculation of the maximum element of an array of RNS numbers using mixed-radix conversion
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
    extern __shared__ mrd_t shared[];
    shared[threadIdx.x].idx = -1;
    while (numberIdx < N){
        if(rns_max_cmp_mrc(&in[numberIdx], &shared[threadIdx.x]) == 1){
            shared[threadIdx.x] = in[numberIdx];
        }
        numberIdx +=  gridDim.x * blockDim.x;
    }
    __syncthreads();
    auto i = cuda::NEXT_POW2(blockDim.x) >> 1;
    while(i >= 1) {
        if ((threadIdx.x < i) && (threadIdx.x + i < blockDim.x) && rns_max_cmp_mrc(&shared[threadIdx.x + i], &shared[threadIdx.x]) == 1) {
            shared[threadIdx.x] = shared[threadIdx.x + i];
        }
        i = i >> 1;
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = shared[threadIdx.x];
}

__global__ void rns_max_mrc_set_kernel(int *out, int *in, mrd_t *mrd){
    int idx = mrd[0].idx * RNS_MODULI_SIZE + threadIdx.x;
    out[threadIdx.x] = in[idx];
}

void rns_max_mrc(int *out, int *in, unsigned int N, mrd_t *buffer) {
    //Execution config
    int gridDim1 = 128;
    int blockDim1 = 64;
    int gridDim2 = 256;
    int blockDim2 = 128;

    size_t memSize = blockDim2 * sizeof(mrd_t);
    rns_max_mrc_compute_kernel <<< gridDim1, blockDim1 >>> ( buffer, in, N);
    //rns_max_mrc_compute_kernel <<< 4096, 64 >>> ( buffer, in, N);
    rns_max_mrc_tree_kernel <<< gridDim2, blockDim2, memSize >>> (buffer, buffer, N);
    rns_max_mrc_tree_kernel <<< 1, blockDim2, memSize >>> (buffer, buffer, gridDim2);
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
    //Device data
    int * dresult;
    xinterval_t * dbuf;
    cudaMalloc(&dresult, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&dbuf, sizeof(xinterval_t) * array_size);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    cuda::rns_max<
            RNS_MAX_NUM_BLOCKS_1,
            RNS_MAX_BLOCK_SIZE_1,
            RNS_MAX_NUM_BLOCKS_2,
            RNS_MAX_BLOCK_SIZE_2>(dresult, drx, array_size, dbuf);
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
    //Device data
    int * dresult;
    mrd_t * dbuf;
    cudaMalloc(&dresult, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&dbuf, sizeof(mrd_t) * array_size);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    rns_max_mrc(dresult, drx, array_size, dbuf);
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
    fill_random_array(hx, array_size, BND_RNS_MODULI_PRODUCT, false);
    //Convert to the RNS
    for(int i = 0; i < array_size; i++){
        rns_from_binary(&hrx[i*RNS_MODULI_SIZE], hx[i]);
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
    Logger::printDash();
    Logger::printParam("RNS_MAX_NUM_BLOCKS_1", RNS_MAX_NUM_BLOCKS_1);
    Logger::printParam("RNS_MAX_BLOCK_SIZE_1", RNS_MAX_BLOCK_SIZE_1);
    Logger::printParam("RNS_MAX_NUM_BLOCKS_2", RNS_MAX_NUM_BLOCKS_2);
    Logger::printParam("RNS_MAX_BLOCK_SIZE_2", RNS_MAX_BLOCK_SIZE_2);
    Logger::endSection(true);
    Logger::printSpace();
    run_test(ARRAY_SIZE);
    Logger::endTestDescription();
    return 0;
}