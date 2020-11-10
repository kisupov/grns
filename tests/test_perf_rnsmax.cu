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
#define RNS_MAX_NUM_BLOCKS_1 8192
#define RNS_MAX_BLOCK_SIZE_1 64
#define RNS_MAX_NUM_BLOCKS_2 1024
#define RNS_MAX_BLOCK_SIZE_2 64

static void printResult(int * result){
    mpz_t binary;
    mpz_init(binary);
    rns_to_binary(binary, result);
    printf("result: %s", mpz_get_str(NULL, 10, binary));
    mpz_clear(binary);
}

void resetResult(int * r){
    memset(r, 0, RNS_MODULI_SIZE * sizeof(int));
}

__global__ void resetResultCuda(int * r) {
  for(int i = 0; i < RNS_MODULI_SIZE; i++){
      r[i] = 0;
  }
}

void run_test(int array_size){
    InitCudaTimer();

    mpz_t * hx = new mpz_t[array_size];
    mpz_t hmax;

    //Host data
    int * hrx = new int[array_size * RNS_MODULI_SIZE];
    int hrmax[RNS_MODULI_SIZE];

    //GPU data
    int * drx;
    int * drmax;
    xinterval_t * dbuf;

    //Memory allocation
    for(int i = 0; i < array_size; i++){
        mpz_init(hx[i]);
    }
    mpz_init(hmax);

    cudaMalloc(&drx, sizeof(int) * array_size * RNS_MODULI_SIZE);
    cudaMalloc(&drmax, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&dbuf, sizeof(xinterval_t) * array_size);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Generate inputs
    fill_random_array(hx, array_size, BND_RNS_MODULI_PRODUCT, false);

    //Convert to the RNS
    for(int i = 0; i < array_size; i++){
        rns_from_binary(&hrx[i*RNS_MODULI_SIZE], hx[i]);
    }

    // Copying to the GPU
    cudaMemcpy(drx, hrx, sizeof(int) * array_size * RNS_MODULI_SIZE, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Compute exact result
    //---------------------------------------------------------
    Logger::printDash();
    mpz_set_ui(hmax, 0);
    for(int i = 0; i < array_size; i++){
        if(mpz_cmp(hx[i], hmax) > 0){
            mpz_set(hmax, hx[i]);
        }
    }
    printf("[GNU MP] max: \nresult: %s", mpz_get_str(NULL, 10, hmax));
    Logger::printSpace();
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] rns_max");
    resetResult(hrmax);
    resetResultCuda<<<1, 1>>>(drmax);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    cuda::rns_max<
            RNS_MAX_NUM_BLOCKS_1,
            RNS_MAX_BLOCK_SIZE_1,
            RNS_MAX_NUM_BLOCKS_2,
            RNS_MAX_BLOCK_SIZE_2>(drmax, drx, array_size, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");
    //Copying to the host
    cudaMemcpy(hrmax, drmax, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    printResult(hrmax);
    Logger::printSpace();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //---------------------------------------------------------
    Logger::printSpace();
    //Cleanup
    for(int i = 0; i < array_size; i++){
        mpz_clear(hx[i]);
    }
    mpz_clear(hmax);
    delete [] hx;
    delete [] hrx;
    cudaFree(drx);
    cudaFree(drmax);
    cudaFree(dbuf);
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