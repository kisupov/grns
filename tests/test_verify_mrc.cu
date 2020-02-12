/*
 * Test for checking the correctness of the mixed-radix conversion routines
 */

#include <iostream>
#include "gmp.h"
#include "logger.cuh"
#include "../src/mrc.cuh"

#define ITERATIONS 30

/*
 * Some common routines
 */

static void inc_rns(int * number){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        number[i] = (number[i] + 1) % RNS_MODULI[i];
    }
}

static void dec_rns(int * number){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        number[i] = number[i] - 1;
        if(number[i] < 0)
            number[i] = number[i] + RNS_MODULI[i];
    }
}

static void clear_mrs(int * number){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        number[i] = 0;
    }
}

__global__ void clear_device_mrs(int * dev_number) {
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        dev_number[i] = 0;
    }
}

/*
 * Methods for printing test results
 */

void print_mrs(int * number) {
    for (int i = 0; i < RNS_MODULI_SIZE; i++)
        printf("%i ", number[i]);
}

static void print_test_result(const char* name, int * mrc_result){
    mpz_t binary;
    mpz_init(binary);
    std::cout<<std::endl<<name;
    printf("\n\tMRS: ");
    print_mrs(mrc_result);
    mrs_to_binary(binary, mrc_result);
    printf("\n\tBIN: %s\n", mpz_get_str(NULL, 10, binary));
    mpz_clear(binary);
}

/*
 * GPU tests
 */
__global__ void run_serial_mrc(int * dev_number, int * dev_mrc_result) {
    cuda::perform_mrc(dev_mrc_result, dev_number);
}

__global__ void run_parallel_mrc_thread(int * dev_number, int * dev_mrc_result) {
    cuda::perform_mrc_parallel(dev_mrc_result, dev_number);
}

void make_iteration(int * number) {
    mpz_t binary;
    mpz_init(binary);
    printf("\n");
    Logger::printDash();
    printf("Original number:");
    rns_to_binary(binary, number);
    printf("\n\tBIN: %s\n", mpz_get_str(NULL, 10, binary));
    mpz_clear(binary);

    /*
     * CPU test
     */
    int mrc_result[RNS_MODULI_SIZE];
    perform_mrc(mrc_result, number);
    print_test_result("[CPU] Szabo-Tanaka MRC:", mrc_result);
    clear_mrs(mrc_result);

    /*
     * GPU test
     */
    int * dev_number;
    int * dev_mrc_result;
    // Memory allocation
    cudaMalloc(&dev_number, sizeof(int) * RNS_MODULI_SIZE);
    cudaMalloc(&dev_mrc_result, sizeof(int) * RNS_MODULI_SIZE);
    // Copying the inputs
    cudaMemcpy(dev_number, number, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyHostToDevice);

    //Call the Szabo-Tanaka kernel
    run_serial_mrc << < 1, 1 >> > (dev_number, dev_mrc_result);
    cudaDeviceSynchronize();
    // Copying the results back to host
    cudaMemcpy(mrc_result, dev_mrc_result, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    // Printing the results
    print_test_result("[CUDA] Szabo-Tanaka MRC:", mrc_result);
    clear_mrs(mrc_result);
    clear_device_mrs<< < 1, 1 >> > (dev_mrc_result);

    //Call the Gbolagade-Cotofana kernel (thread)
    run_parallel_mrc_thread<< < 1, RNS_MODULI_SIZE >>> (dev_number, dev_mrc_result);
    cudaDeviceSynchronize();
    // Copying the results back to host
    cudaMemcpy(mrc_result, dev_mrc_result, sizeof(int) * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    // Printing the results
    print_test_result("[CUDA] Gbolagade-Cotofana Thread MRC:", mrc_result);
    clear_mrs(mrc_result);
    clear_device_mrs<< < 1, 1 >> > (dev_mrc_result);

    cudaFree(dev_number);
    cudaFree(dev_mrc_result);
    cudaDeviceSynchronize();
}


int main() {
    Logger::beginTestDescription(Logger::TEST_VERIFY_MRC);
    Logger::printSpace();
    rns_const_init();

    bool asc = false;
    int number[RNS_MODULI_SIZE];

    if(asc){
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            number[i] = 0;
        }
        for (int i = 0; i < ITERATIONS; i++) {
            inc_rns(number);
            make_iteration(number);
        }
    } else{
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            number[i] = RNS_MODULI[i];
        }
        for (int i = 0; i < ITERATIONS; i++) {
            dec_rns(number);
            make_iteration(number);
        }
    }
    Logger::endTestDescription();
    return 1;
}