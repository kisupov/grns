/*
 *  Test for checking the RNS division algorithms
 */

#include <stdio.h>
#include <iostream>
#include "../src/rnsdiv.cuh"
#include "tsthelper.cuh"
#include "logger.cuh"

static void printResult(int * quotient, int * remainder){
    mpz_t binary;
    mpz_init(binary);
    rns_to_binary(binary, quotient);
    printf("\nquotient:  %s", mpz_get_str(NULL, 10, binary));
    rns_to_binary(binary, remainder);
    printf("\nremainder: %s", mpz_get_str(NULL, 10, binary));
    mpz_clear(binary);
}

void resetResult(int * q, int * r){
    memset(q, 0, RNS_MODULI_SIZE * sizeof(int));
    memset(r, 0, RNS_MODULI_SIZE * sizeof(int));
}

__global__ void resetResultCuda(int * q, int * r) {
  for(int i = 0; i < RNS_MODULI_SIZE; i++){
      q[i] = 0;
      r[i] = 0;
  }
}

__global__ void testCudaRnsDiv(int * dq, int * dr, int * dx, int * dd) {
    cuda::rns_div(dq, dr, dx, dd);
}

__global__ void testCudaRnsDivParallel(int * dq, int * dr, int * dx, int * dd) {
    cuda::rns_div_parallel(dq, dr, dx, dd);
}

__global__ void testCudaRnsDivParallelFast(int * dq, int * dr, int * dx, int * dd) {
    cuda::rns_div_parallel_fast(dq, dr, dx, dd);
}

void make_iteration(){
    mpz_t hx;
    mpz_t hd;
    mpz_t hq;
    mpz_t hr;

    //Host data
    int hrx[RNS_MODULI_SIZE];
    int hrd[RNS_MODULI_SIZE];
    int hrq[RNS_MODULI_SIZE];
    int hrr[RNS_MODULI_SIZE];

    //GPU data
    int * drx;
    int * drd;
    int * drq;
    int * drr;

    //Memory allocation
    mpz_init(hx);
    mpz_init(hd);
    mpz_init(hq);
    mpz_init(hr);

    cudaMalloc(&drx, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&drd, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&drq, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&drr, sizeof(int) * RNS_MODULI_SIZE);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Generate inputs
    fill_random_array(&hx, 1, BND_RNS_MODULI_PRODUCT, false);
    fill_random_array(&hd, 1, BND_RNS_MODULI_PRODUCT_SQRT, false);

    //Convert to the RNS
    rns_from_binary(hrx, hx);
    rns_from_binary(hrd, hd);

    // Copying to the GPU
    cudaMemcpy(drx, hrx, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(drd, hrd, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Compute exact result
    //---------------------------------------------------------
    Logger::printDash();
    mpz_div(hq, hx, hd);
    mpz_mod(hr, hx, hd);
    printf("dividend:  %s", mpz_get_str(NULL, 10, hx));
    printf("\ndivisor:   %s", mpz_get_str(NULL, 10, hd));
    printf("\nquotient:  %s", mpz_get_str(NULL, 10, hq));
    printf("\nremainder: %s", mpz_get_str(NULL, 10, hr));
    Logger::printSpace();
    //---------------------------------------------------------
    Logger::printDash();
    printf("[CPU] rns_div:");
    resetResult(hrq, hrr);
    rns_div(hrq, hrr, hrx, hrd);
    printResult(hrq, hrr);
    Logger::printSpace();
    //---------------------------------------------------------
    Logger::printDash();
    printf("[CPU] rns_div_fast:");
    resetResult(hrq, hrr);
    rns_div_fast(hrq, hrr, hrx, hrd);
    printResult(hrq, hrr);
    Logger::printSpace();
    //---------------------------------------------------------
    Logger::printDash();
    printf("[CUDA] rns_div:");
    resetResult(hrq, hrr);
    resetResultCuda<<<1, 1>>>(drq, drr);
    testCudaRnsDiv<<<1,1>>>(drq, drr, drx, drd);
    // Copying the results back to host
    cudaMemcpy(hrq, drq, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrr, drr, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    printResult(hrq, hrr);
    Logger::printSpace();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //---------------------------------------------------------
    Logger::printDash();
    printf("[CUDA] rns_div_parallel:");
    resetResult(hrq, hrr);
    resetResultCuda<<<1, 1>>>(drq, drr);
    testCudaRnsDivParallel<<<1,RNS_MODULI_SIZE>>>(drq, drr, drx, drd);
    // Copying the results back to host
    cudaMemcpy(hrq, drq, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrr, drr, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    printResult(hrq, hrr);
    Logger::printSpace();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //---------------------------------------------------------
    Logger::printDash();
    printf("[CUDA] rns_div_parallel_fast:");
    resetResult(hrq, hrr);
    resetResultCuda<<<1, 1>>>(drq, drr);
    testCudaRnsDivParallelFast<<<1,RNS_MODULI_SIZE>>>(drq, drr, drx, drd);
    // Copying the results back to host
    cudaMemcpy(hrq, drq, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrr, drr, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    printResult(hrq, hrr);
    Logger::printSpace();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //---------------------------------------------------------
    Logger::printSpace();
    //Cleanup
    mpz_clear(hx);
    mpz_clear(hd);
    mpz_clear(hq);
    mpz_clear(hr);
    cudaFree(drx);
    cudaFree(drd);
    cudaFree(drq);
    cudaFree(drr);
}

int main() {
    rns_const_init();
    Logger::beginTestDescription(Logger::TEST_VERIFY_RNSDIV);
    rns_const_print(true);
    Logger::printDash();
    rns_eval_const_print();
    Logger::endSection(true);
    Logger::printSpace();
    //Launch
    make_iteration();
    //End logging
    Logger::printSpace();
    Logger::endTestDescription();
    return 1;
}