/*
 *  Test for measure the performance of the various multiple-precision integer addition algorithms (naive vs optimized)
 */

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "gmp.h"
#include "../src/mpint.cuh"
#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"

#define ITERATIONS 1000000


/*
 * Addition of multiple-precision numbers, result = x + y.
 * The signs of the operands are ignored in this algorithm.
 * No overflow check is performed
 */
DEVICE_CUDA_FORCEINLINE void mpint_add_naive_unsign(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
    int moduli[RNS_MODULI_SIZE];
    int dig[RNS_MODULI_SIZE];
    int digx[RNS_MODULI_SIZE];
    int digy[RNS_MODULI_SIZE];
    for(int i = 0; i < RNS_MODULI_SIZE; i ++){
        moduli[i] = cuda::RNS_MODULI[i];
        digx[i] = x->digits[i];
        digy[i] = y->digits[i];
    }
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        dig[i] = cuda::mod_add(digx[i], digy[i], moduli[i]);
    }
    cuda::er_add_rd(&result->eval[0], &x->eval[0], &y->eval[0]);
    cuda::er_add_ru(&result->eval[1], &x->eval[1], &y->eval[1]);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        result->digits[i] = dig[i];
    }
}

/*
 * Subtraction of multiple-precision numbers, result = x - y, x >= y
 * The signs of the operands are ignored in this algorithm.
 * No overflow check is performed
 */
DEVICE_CUDA_FORCEINLINE void mpint_sub_naive_unsign(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
    int moduli[RNS_MODULI_SIZE];
    int dig[RNS_MODULI_SIZE];
    int digx[RNS_MODULI_SIZE];
    int digy[RNS_MODULI_SIZE];
    for(int i = 0; i < RNS_MODULI_SIZE; i ++){
        moduli[i] = cuda::RNS_MODULI[i];
        digx[i] = x->digits[i];
        digy[i] = y->digits[i];
    }
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        dig[i] = cuda::mod_psub(digx[i], digy[i], moduli[i]);
    }
    cuda::er_sub_rd(&result->eval[0], &x->eval[0], &y->eval[1]);
    cuda::er_sub_ru(&result->eval[1], &x->eval[1], &y->eval[0]);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        result->digits[i] = dig[i];
    }
}

/*
 * Naive CUDA implementation of multiple-precision addition, result = x + y
 * Checking the signs of the operands via if–else statements results in branch divergence in this algorithm
 * For simplicity, negative zero is permitted, no overflow check is performed
 */
DEVICE_CUDA_FORCEINLINE void mpint_add_naive(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
    int sx = x->sign;
    int sy = y->sign;
    if(sx == sy){
        result->sign = sx;
        mpint_add_naive_unsign(result, x, y);
    } else if(cuda::er_ucmp(&x->eval[0], &y->eval[1]) >= 0){
        result->sign = sx;
        mpint_sub_naive_unsign(result, x, y);
    } else if(cuda::er_ucmp(&y->eval[0], &x->eval[1]) >= 0){
        result->sign = sy;
        mpint_sub_naive_unsign(result, y, x);
    }else{
        int cmp = cuda::mrc_compare_rns(x->digits, y->digits);
        if(cmp >=0){
            result->sign = sx;
            cuda::rns_sub(result->digits, x->digits, y->digits);
        } else{
            result->sign = sy;
            cuda::rns_sub(result->digits, y->digits, x->digits);
        }
        cuda::rns_eval_compute(&result->eval[0], &result->eval[1],result->digits);
    }
}

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

__global__ static void testCudaMpAddNaive(mp_int_t * dz, mp_int_t * dx, mp_int_t * dy, int vectorSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < vectorSize) {
        mpint_add_naive(&dz[idx],&dx[idx],&dy[idx]);
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
static void run_test(int iterations, RandomBoundType randomBoundType, bool allowNegative, bool allNegative) {
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
    fill_random_array(mpzx, iterations, randomBoundType, allowNegative);
    waitFor(5);
    fill_random_array(mpzy, iterations, randomBoundType, allowNegative);

    //The generated data is assumed to be positive
    if(allNegative){
        #pragma omp parallel for
        for(int i = 0; i < iterations; i++){
            mpz_mul_si(mpzx[i], mpzx[i], -1);
            mpz_mul_si(mpzy[i], mpzy[i], -1);
        }
    }

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
    // MPINT CUDA add naive testing
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] GRNS mpint_add_naive");
    resetResult(hz, iterations);
    resetResultCuda<<<blocks,threads>>>(dz, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    testCudaMpAddNaive<<<blocks,threads>>>(dz, dx, dy, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    //Copying to the host
    cudaMemcpy(hz, dz, sizeof(mp_int_t) * iterations , cudaMemcpyDeviceToHost);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    checkResult(mpzz, hz, iterations);
    //---------------------------------------------------------

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
    Logger::printParam("PRECISION", RNS_MODULI_PRODUCT_LOG2);
    Logger::printDash();
    rns_const_print(true);
    Logger::endSection(true);
    run_test(ITERATIONS, BND_RNS_MODULI_PRODUCT_HALF, false, false);
    Logger::endTestDescription();
    return 0;
}