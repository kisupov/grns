/*
 *  Test for checking the routine for finding the maximum element in an array of RNS numbers
 */

#include <stdio.h>
#include <iostream>
#include "../src/dpp/rnsmax.cuh"
#include "tsthelper.cuh"
#include "logger.cuh"

#define RNS_MAX_NUM_BLOCKS_1 32
#define RNS_MAX_BLOCK_SIZE_1 32
#define RNS_MAX_NUM_BLOCKS_2 32
#define RNS_MAX_BLOCK_SIZE_2 32

static void printResult(int * result){
    mpz_t binary;
    mpz_init(binary);
    rns_to_binary(binary, result);
    printf("\nresult: %s", mpz_get_str(NULL, 10, binary));
    mpz_clear(binary);
}

void run_test(int array_size){
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
    printf("[CUDA] rns_max:");
    cuda::rns_max<
            RNS_MAX_NUM_BLOCKS_1,
            RNS_MAX_BLOCK_SIZE_1,
            RNS_MAX_NUM_BLOCKS_2,
            RNS_MAX_BLOCK_SIZE_2>(drmax, drx, array_size, dbuf);
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
    rns_const_init();
    Logger::beginTestDescription(Logger::TEST_VERIFY_RNSMAX);
    rns_const_print(true);
    Logger::printDash();
    rns_eval_const_print();
    Logger::endSection(true);
    Logger::printSpace();
    //Launch
    run_test(10000);
    //End logging
    Logger::printSpace();
    Logger::endTestDescription();
    return 1;
}