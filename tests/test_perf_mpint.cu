/*
 *  Test for measure the performance of the multiple-precision integer routines
 */

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "gmp.h"
#include "../src/mpint.cuh"
#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"

enum mpintTestType {
    add_test,
    sub_test,
    mul_test,
    div_test
};

#define ITERATIONS 1000000

/*
 * CUDA tests
 */

__global__ static void testCudaMpAdd(mp_int_t * dz, mp_int_t * dx, mp_int_t * dy, int vectorSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < vectorSize) {
        cuda::mpint_add(&dz[idx],&dx[idx],&dy[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

__global__ static void testCudaMpSub(mp_int_t * dz, mp_int_t * dx, mp_int_t * dy, int vectorSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < vectorSize) {
        cuda::mpint_sub(&dz[idx],&dx[idx],&dy[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

__global__ static void testCudaMpMul(mp_int_t * dz, mp_int_t * dx, mp_int_t * dy, int vectorSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < vectorSize) {
        cuda::mpint_mul(&dz[idx],&dx[idx],&dy[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

__global__ static void testCudaMpDiv(mp_int_t * dz, mp_int_t * dx, mp_int_t * dy, int vectorSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < vectorSize) {
        cuda::mpint_div(&dz[idx],&dx[idx],&dy[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

/*
 * Common methods
 */

static void resetResult(mp_int_t * r, int vectorSize){
    #pragma omp parallel for
    for(auto i = 0; i < vectorSize; i++){
        mpint_set_i(&r[i], 0);
    }
}

static void resetResult(mpz_t * r, int vectorSize){
    #pragma omp parallel for
    for(auto i = 0; i < vectorSize; i++){
        mpz_set_ui(r[i], 0);
    }
}

__global__ static void resetResultCuda(mp_int_t * r, int vectorSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < vectorSize) {
       cuda::mpint_set(&r[idx], &cuda::MPINT_ZERO);

        idx += gridDim.x * blockDim.x;
    }
}

static void checkResult(mpz_t * ref, mp_int_t * res, int vectorSize){
    int errors = 0;
    mpz_t mp;
    mpz_init(mp);
    for(int i = 0; i < vectorSize; i++){
        mpint_get_mpz(mp, &res[i]);
        if(mpz_cmp(mp, ref[i]) != 0){
            errors++;
        }
    }
    if(errors == 0){
        printf("All results match\n");
    }else{
        printf("Count of errors: %i\n", errors);
    }
    mpz_clear(mp);
}

/*
 * Main test
 */
static void run_test(int iterations, mpintTestType testType, RandomBoundType randomBoundType) {
    InitCpuTimer();
    InitCudaTimer();

    //Execution configuration
    int threads = 32;
    int blocks = iterations / threads + (iterations % threads ? 1 : 0);

    // Multiple-precision GMP host data
    mpz_t * mpzx  = new mpz_t[iterations];
    mpz_t * mpzy  = new mpz_t[iterations];
    mpz_t * mpzz  = new mpz_t[iterations];

    // Multiple-precision mp_int host data
    auto *hx = new mp_int_t[iterations];
    auto *hy = new mp_int_t[iterations];
    auto *hz = new mp_int_t[iterations];

    //GPU data
    mp_int_t * dx;
    mp_int_t * dy;
    mp_int_t * dz;

    //Memory allocation
    for(int i = 0; i < iterations; i++){
        mpz_init(mpzx[i]);
        mpz_init(mpzy[i]);
        mpz_init(mpzz[i]);
    }

    cudaMalloc(&dx, sizeof(mp_int_t) * iterations);
    cudaMalloc(&dy, sizeof(mp_int_t) * iterations);
    cudaMalloc(&dz, sizeof(mp_int_t) * iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Generate inputs
    fill_random_array(mpzx, iterations, randomBoundType, false);
    waitFor(5);
    fill_random_array(mpzy, iterations, randomBoundType, false);

    //Convert to the RNS
    #pragma omp parallel for
    for(int i = 0; i < iterations; i++){
        mpint_set_mpz(&hx[i], mpzx[i]);
        mpint_set_mpz(&hy[i], mpzy[i]);
    }

    // Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(mp_int_t) * iterations, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, sizeof(mp_int_t) * iterations, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    switch (testType)
    {
        case add_test:
            //---------------------------------------------------------
            // GMP add testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CPU] GNU MP mpz_add");
            resetResult(mpzz, iterations);
            StartCpuTimer();
            #pragma omp parallel for
            for(int i = 0; i < iterations; i++){
                mpz_add(mpzz[i],mpzx[i],mpzy[i]);
            }
            EndCpuTimer();
            PrintCpuTimer("took");
            //---------------------------------------------------------
            // MPINT add testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CPU] GRNS mpint_add");
            resetResult(hz, iterations);
            StartCpuTimer();
            #pragma omp parallel for
            for(int i = 0; i < iterations; i++){
                mpint_add(&hz[i],&hx[i],&hy[i]);
            }
            EndCpuTimer();
            PrintCpuTimer("took");
            checkResult(mpzz, hz, iterations);
            //---------------------------------------------------------
            // MPINT CUDA add testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CUDA] GRNS mpint_add");
            resetResult(hz, iterations);
            resetResultCuda<<<blocks,threads>>>(dz, iterations);
            checkDeviceHasErrors(cudaDeviceSynchronize());
            cudaCheckErrors();
            //Launch
            StartCudaTimer();
            testCudaMpAdd<<<blocks,threads>>>(dz, dx, dy, iterations);
            EndCudaTimer();
            PrintCudaTimer("took");
            //Copying to the host
            cudaMemcpy(hz, dz, sizeof(mp_int_t) * iterations , cudaMemcpyDeviceToHost);
            checkDeviceHasErrors(cudaDeviceSynchronize());
            cudaCheckErrors();
            checkResult(mpzz, hz, iterations);
            //---------------------------------------------------------
            break;
        case sub_test:
            //---------------------------------------------------------
            // GMP sub testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CPU] GNU MP mpz_sub");
            resetResult(mpzz, iterations);
            StartCpuTimer();
            #pragma omp parallel for
            for(int i = 0; i < iterations; i++){
                mpz_sub(mpzz[i],mpzx[i],mpzy[i]);
            }
            EndCpuTimer();
            PrintCpuTimer("took");
            //---------------------------------------------------------
            // MPINT sub testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CPU] GRNS mpint_sub");
            resetResult(hz, iterations);
            StartCpuTimer();
            #pragma omp parallel for
            for(int i = 0; i < iterations; i++){
                mpint_sub(&hz[i],&hx[i],&hy[i]);
            }
            EndCpuTimer();
            PrintCpuTimer("took");
            checkResult(mpzz, hz, iterations);
            //---------------------------------------------------------
            // MPINT CUDA sub testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CUDA] GRNS mpint_sub");
            resetResult(hz, iterations);
            resetResultCuda<<<blocks,threads>>>(dz, iterations);
            checkDeviceHasErrors(cudaDeviceSynchronize());
            cudaCheckErrors();
            //Launch
            StartCudaTimer();
            testCudaMpSub<<<blocks,threads>>>(dz, dx, dy, iterations);
            EndCudaTimer();
            PrintCudaTimer("took");
            //Copying to the host
            cudaMemcpy(hz, dz, sizeof(mp_int_t) * iterations , cudaMemcpyDeviceToHost);
            checkDeviceHasErrors(cudaDeviceSynchronize());
            cudaCheckErrors();
            checkResult(mpzz, hz, iterations);
            //---------------------------------------------------------
            break;
        case mul_test:
            //---------------------------------------------------------
            // GMP mul testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CPU] GNU MP mpz_mul");
            resetResult(mpzz, iterations);
            StartCpuTimer();
            #pragma omp parallel for
            for(int i = 0; i < iterations; i++){
                mpz_mul(mpzz[i],mpzx[i],mpzy[i]);
            }
            EndCpuTimer();
            PrintCpuTimer("took");
            //---------------------------------------------------------
            // MPINT mul testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CPU] GRNS mpint_mul");
            resetResult(hz, iterations);
            StartCpuTimer();
            #pragma omp parallel for
            for(int i = 0; i < iterations; i++){
                mpint_mul(&hz[i],&hx[i],&hy[i]);
            }
            EndCpuTimer();
            PrintCpuTimer("took");
            checkResult(mpzz, hz, iterations);
            //---------------------------------------------------------
            // MPINT CUDA mul testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CUDA] GRNS mpint_mul");
            resetResult(hz, iterations);
            resetResultCuda<<<blocks,threads>>>(dz, iterations);
            checkDeviceHasErrors(cudaDeviceSynchronize());
            cudaCheckErrors();
            //Launch
            StartCudaTimer();
            testCudaMpMul<<<blocks,threads>>>(dz, dx, dy, iterations);
            EndCudaTimer();
            PrintCudaTimer("took");
            //Copying to the host
            cudaMemcpy(hz, dz, sizeof(mp_int_t) * iterations , cudaMemcpyDeviceToHost);
            checkDeviceHasErrors(cudaDeviceSynchronize());
            cudaCheckErrors();
            checkResult(mpzz, hz, iterations);
            break;
        case div_test:
            //---------------------------------------------------------
            // GMP div testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CPU] GNU MP mpz_fdiv_q");
            resetResult(mpzz, iterations);
            StartCpuTimer();
            #pragma omp parallel for
            for(int i = 0; i < iterations; i++){
                mpz_fdiv_q(mpzz[i],mpzx[i],mpzy[i]);
            }
            EndCpuTimer();
            PrintCpuTimer("took");
            //---------------------------------------------------------
            // MPINT div testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CPU] GRNS mpint_div");
            resetResult(hz, iterations);
            StartCpuTimer();
            #pragma omp parallel for
            for(int i = 0; i < iterations; i++){
                mpint_div(&hz[i],&hx[i],&hy[i]);
            }
            EndCpuTimer();
            PrintCpuTimer("took");
            checkResult(mpzz, hz, iterations);
            //---------------------------------------------------------
            // MPINT CUDA div testing
            //---------------------------------------------------------
            Logger::printDash();
            PrintTimerName("[CUDA] GRNS mpint_div");
            resetResult(hz, iterations);
            resetResultCuda<<<blocks,threads>>>(dz, iterations);
            checkDeviceHasErrors(cudaDeviceSynchronize());
            cudaCheckErrors();
            //Launch
            StartCudaTimer();
            testCudaMpDiv<<<blocks,threads>>>(dz, dx, dy, iterations);
            EndCudaTimer();
            PrintCudaTimer("took");
            //Copying to the host
            cudaMemcpy(hz, dz, sizeof(mp_int_t) * iterations , cudaMemcpyDeviceToHost);
            checkDeviceHasErrors(cudaDeviceSynchronize());
            cudaCheckErrors();
            checkResult(mpzz, hz, iterations);
            break;
        default:
            break;
    }

    // Cleanup
    #pragma omp parallel for
    for(int i = 0; i < iterations; i++){
        mpz_clear(mpzx[i]);
        mpz_clear(mpzy[i]);
        mpz_clear(mpzz[i]);
    }
    delete [] hx;
    delete [] hy;
    delete [] hz;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
}

int main() {
    cudaDeviceReset();
    rns_const_init();
    mpint_const_init();
    Logger::beginTestDescription(Logger::TEST_PERF_MPINT);
    Logger::printParam("ITERATIONS", ITERATIONS);
    Logger::printDash();
    rns_const_print(true);
    Logger::endSection(true);
    run_test(ITERATIONS, add_test, BND_RNS_MODULI_PRODUCT_HALF);
    Logger::printSpace();
    run_test(ITERATIONS, sub_test, BND_RNS_MODULI_PRODUCT_HALF);
    Logger::printSpace();
    run_test(ITERATIONS, mul_test, BND_RNS_MODULI_PRODUCT_SQRT);
    Logger::printSpace();
    run_test(ITERATIONS, div_test, BND_RNS_MODULI_PRODUCT);
    Logger::endTestDescription();
    return 0;
}