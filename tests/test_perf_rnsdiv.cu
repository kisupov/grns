/*
 *  Test for measure the performance of the RNS division algorithms
 */

#include <stdio.h>
#include <iostream>
#include "gmp.h"
#include "../src/rnsdiv.cuh"
#include "3rdparty/campary/Doubles/src_gpu/multi_prec.h"
#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"


#define ITERATIONS 1000

/***
 * The RNS division algorithm proposed by Hiasat and Abdel-Aty-Zohdy
 * Ahmad A. Hiasat, Hoda S. Abdel-Aty-Zohdy. Design and Implementation of An RNS Division Algorithm
 * Parallel implementation of the Realization II
 ***/
#define RNS_MODULI_SIZE_LOG2 4
#define CAMPARY_PRECISION (RNS_MODULI_PRODUCT_LOG2 + RNS_MODULI_SIZE_LOG2) %53 ? \
(int)((RNS_MODULI_PRODUCT_LOG2 + RNS_MODULI_SIZE_LOG2)/53 + 1) : (RNS_MODULI_PRODUCT_LOG2 + RNS_MODULI_SIZE_LOG2)/53
typedef multi_prec<CAMPARY_PRECISION> bignum_t;

/*
 * Routine that extracts the most-significant bit of bignum_t x. It should be equivalent with floor(log_2(x)) +/- 1
 */
DEVICE_CUDA_FORCEINLINE static int msb(bignum_t x) {
    //We assume that the input is a number represented as a set of non-overlapped floating-point expansions,
    //so the exponent of x.getData()[0] gives the correct result.
    RealIntUnion u;
    u.dvalue = x.getData()[0];
    //Zeroing the sign and displacing the significand by the exponent value. If the significand is zero, then the exponent is also zeroed.
    return (int)(((u.ivalue & ~((uint64_t) 1 << DBL_SIGN_OFFSET)) >> DBL_EXP_OFFSET) - DBL_EXP_BIAS) * (x.getData()[0] != 0);
}

/*
 * Routine that computes the most-significant bit of an RNS number x, i.e. h(x), via full precision evaluation
 * of the fractional representation. CAMPARY is used for multiple-precision arithmetic
 */
DEVICE_CUDA_FORCEINLINE static int higherPow(int * x){
    //Computing the full-precision fractional representation
    __shared__ bignum_t s[RNS_MODULI_SIZE];
    bignum_t piece = (double)cuda::mod_mul(x[threadIdx.x], cuda::RNS_PART_MODULI_PRODUCT_INVERSE[threadIdx.x], cuda::RNS_MODULI[threadIdx.x]);
    divExpans_d<CAMPARY_PRECISION, CAMPARY_PRECISION>(s[threadIdx.x], piece, (double)cuda::RNS_MODULI[threadIdx.x]);
    __syncthreads();
    //Parallel reduction
    for (unsigned int i = RNS_PARALLEL_REDUCTION_IDX; i > 0; i >>= 1) {
        if (threadIdx.x < i && threadIdx.x + i < RNS_MODULI_SIZE) {
            s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + i];
        }
        __syncthreads();
    }
    //Computing the most significant bit of s[0]
    if (threadIdx.x == 0) {
        //Discard the integer part
        for(int i = RNS_MODULI_SIZE - 1; i > 0; i--){
            if(s[0] >= i){
                s[0] -= i;
                break;
            }
        }
    }
    __syncthreads();
    return msb(s[0]);
}

/*
 * Parallel implementation of the Hiasat and Abdel-Aty-Zohdy's RNS division
 */
DEVICE_CUDA_FORCEINLINE static void hiasatDivision(int *q, int *r, int *x, int *d) {
    __shared__ int shared[RNS_MODULI_SIZE];
    int pq; //Partial quotient
    int ld = d[threadIdx.x]; //Digit of divisor
    int modulus = cuda::RNS_MODULI[threadIdx.x];

    //Set the initial values of q and r
    q[threadIdx.x] = 0;
    r[threadIdx.x] = x[threadIdx.x];

    int k = higherPow(d); // most significand non-zero bit in the divisor
    int j = higherPow(r); // most significand non-zero bit in the dividend (remainder)

    //Main division loop
    while (j > k){
        pq = cuda::RNS_POW2[j-k-1][threadIdx.x]; //Fetch the partial quotient, pq, from the LUT
        q[threadIdx.x] = cuda::mod_add(q[threadIdx.x], pq, modulus); //q = q + pq
        pq = cuda::mod_mul(pq, ld, modulus); //pq = d * pq
        r[threadIdx.x] = cuda::mod_psub(r[threadIdx.x], pq, modulus); //r = r - pq
        j = higherPow(r);
    }
    //Final adjustment
   if(j == k){
       shared[threadIdx.x] = cuda::mod_psub(r[threadIdx.x], ld, modulus); //shared = r - d
        k = higherPow(shared); // most significand non-zero bit in the remainder (shared)
        if(k < -1){
            q[threadIdx.x] = cuda::mod_add(q[threadIdx.x], 1, modulus);
            r[threadIdx.x] = cuda::mod_psub(r[threadIdx.x], ld, modulus);
        }
    }
}


/*
 * CUDA tests
 */

__global__ static void testCudaRnsDivParallel(int * dq, int * dr, int * dx, int * dd, int vectorSize) {
    for(int i = 0; i < vectorSize; i++){
        cuda::rns_div_parallel(&dq[i * RNS_MODULI_SIZE], &dr[i * RNS_MODULI_SIZE], &dx[i * RNS_MODULI_SIZE], &dd[i * RNS_MODULI_SIZE]);
    }
}

__global__ static void testCudaRnsDivParallelFast(int * dq, int * dr, int * dx, int * dd, int vectorSize) {
    for(int i = 0; i < vectorSize; i++){
        cuda::rns_div_parallel_fast(&dq[i * RNS_MODULI_SIZE], &dr[i * RNS_MODULI_SIZE], &dx[i * RNS_MODULI_SIZE], &dd[i * RNS_MODULI_SIZE]);
    }
}

__global__ static void testHiasatDivision(int * dq, int * dr, int * dx, int * dd, int vectorSize) {
    for(int i = 0; i < vectorSize; i++){
        hiasatDivision(&dq[i * RNS_MODULI_SIZE], &dr[i * RNS_MODULI_SIZE], &dx[i * RNS_MODULI_SIZE], &dd[i * RNS_MODULI_SIZE]);
    }
}

/*
 * Common methods
 */

static void resetResult(int * q, int * r, int vectorSize){
    memset(q, 0, RNS_MODULI_SIZE * vectorSize * sizeof(int));
    memset(r, 0, RNS_MODULI_SIZE * vectorSize * sizeof(int));
}

__global__ static void resetResultCuda(int * q, int * r, int vectorSize) {
    for(int i = 0; i < RNS_MODULI_SIZE * vectorSize; i++){
        q[i] = 0;
        r[i] = 0;
    }
}

static void checkResult(mpz_t * q, mpz_t * r, int * rq, int * rr, int vectorSize){
    int errors = 0;
    mpz_t chq;
    mpz_t chr;
    mpz_init(chq);
    mpz_init(chr);
    for(int i = 0; i < vectorSize; i++){
        rns_to_binary(chq, &rq[i * RNS_MODULI_SIZE]);
        rns_to_binary(chr, &rr[i * RNS_MODULI_SIZE]);
        if((mpz_cmp(q[i], chq) != 0) || (mpz_cmp(r[i], chr) != 0)){
            errors++;
        }
    }
    if(errors == 0){
        printf("All results match\n");
    }else{
        printf("Count of errors: %i\n", errors);
    }
    mpz_clear(chq);
    mpz_clear(chr);
}


/*
 * Main test
 */
static void run_test(int iterations) {
    InitCpuTimer();
    InitCudaTimer();

    // Multiple-precision host data
    mpz_t * hx = new mpz_t[iterations];
    mpz_t * hd = new mpz_t[iterations];
    mpz_t * hq = new mpz_t[iterations];
    mpz_t * hr = new mpz_t[iterations];

    // RNS host data
    int * hrx = new int[iterations * RNS_MODULI_SIZE];
    int * hrd = new int[iterations * RNS_MODULI_SIZE];
    int * hrq = new int[iterations * RNS_MODULI_SIZE];
    int * hrr = new int[iterations * RNS_MODULI_SIZE];

    //GPU data
    int * drx;
    int * drd;
    int * drq;
    int * drr;

    //Memory allocation
    for(int i = 0; i < iterations; i++){
        mpz_init(hx[i]);
        mpz_init(hd[i]);
        mpz_init(hq[i]);
        mpz_init(hr[i]);
    }

    cudaMalloc(&drx, sizeof(int) * RNS_MODULI_SIZE * iterations);
    cudaMalloc(&drd, sizeof(int) * RNS_MODULI_SIZE * iterations);
    cudaMalloc(&drq, sizeof(int) * RNS_MODULI_SIZE * iterations);
    cudaMalloc(&drr, sizeof(int) * RNS_MODULI_SIZE * iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Generate inputs
    fill_random_array(hx, iterations, BND_RNS_MODULI_PRODUCT);
    waitFor(5);
    fill_random_array(hd, iterations, BND_RNS_MODULI_PRODUCT_SQRT);

    //Convert to the RNS
    for(int i = 0; i < iterations; i++){
        rns_from_binary(&hrx[i * RNS_MODULI_SIZE], hx[i]);
        rns_from_binary(&hrd[i * RNS_MODULI_SIZE], hd[i]);
    }

    // Copying to the GPU
    cudaMemcpy(drx, hrx, sizeof(int) * RNS_MODULI_SIZE * iterations, cudaMemcpyHostToDevice);
    cudaMemcpy(drd, hrd, sizeof(int) * RNS_MODULI_SIZE * iterations, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Computing exact results
    //---------------------------------------------------------
    for(int i = 0; i < iterations; i++){
        mpz_div(hq[i], hx[i], hd[i]);
        mpz_mod(hr[i], hx[i], hd[i]);
    }
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CPU] rns_div");
    resetResult(hrq, hrr, iterations);
    //Launch
    StartCpuTimer();
    for(int i = 0; i < iterations; i++){
        rns_div(&hrq[i * RNS_MODULI_SIZE], &hrr[i * RNS_MODULI_SIZE], &hrx[i * RNS_MODULI_SIZE], &hrd[i * RNS_MODULI_SIZE]);
    }
    EndCpuTimer();
    PrintCpuTimer("took");
    checkResult(hq, hr, hrq, hrr, iterations);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CPU] rns_div_fast");
    resetResult(hrq, hrr, iterations);
    //Launch
    StartCpuTimer();
    for(int i = 0; i < iterations; i++){
        rns_div_fast(&hrq[i * RNS_MODULI_SIZE], &hrr[i * RNS_MODULI_SIZE], &hrx[i * RNS_MODULI_SIZE], &hrd[i * RNS_MODULI_SIZE]);
    }
    EndCpuTimer();
    PrintCpuTimer("took");
    checkResult(hq, hr, hrq, hrr, iterations);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] rns_div_parallel");
    resetResult(hrq, hrr, iterations);
    resetResultCuda<<<1,1>>>(drq, drr, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    testCudaRnsDivParallel<<<1,RNS_MODULI_SIZE>>>(drq, drr, drx, drd, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    //Copying to the host
    cudaMemcpy(hrq, drq, sizeof(int) * iterations * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrr, drr, sizeof(int) * iterations * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    checkResult(hq, hr, hrq, hrr, iterations);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] rns_div_parallel_fast");
    resetResult(hrq, hrr, iterations);
    resetResultCuda<<<1,1>>>(drq, drr, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    testCudaRnsDivParallelFast<<<1,RNS_MODULI_SIZE>>>(drq, drr, drx, drd, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    //Copying to the host
    cudaMemcpy(hrq, drq, sizeof(int) * iterations * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrr, drr, sizeof(int) * iterations * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    checkResult(hq, hr, hrq, hrr, iterations);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] Hiasat and Abdel-Aty-Zohdy's division");
    resetResult(hrq, hrr, iterations);
    resetResultCuda<<<1,1>>>(drq, drr, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    testHiasatDivision<<<1,RNS_MODULI_SIZE>>>(drq, drr, drx, drd, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    //Copying to the host
    cudaMemcpy(hrq, drq, sizeof(int) * iterations * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrr, drr, sizeof(int) * iterations * RNS_MODULI_SIZE, cudaMemcpyDeviceToHost);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    checkResult(hq, hr, hrq, hrr, iterations);
    //---------------------------------------------------------

    // Cleanup
    for(int i = 0; i < iterations; i++){
        mpz_clear(hx[i]);
        mpz_clear(hd[i]);
        mpz_clear(hq[i]);
        mpz_clear(hr[i]);
    }
    delete [] hx;
    delete [] hd;
    delete [] hq;
    delete [] hr;
    delete [] hrx;
    delete [] hrd;
    delete [] hrq;
    delete [] hrr;
    cudaFree(drx);
    cudaFree(drd);
    cudaFree(drq);
    cudaFree(drr);
}

int main() {
    cudaDeviceReset();
    rns_const_init();
    Logger::beginTestDescription(Logger::TEST_PERF_RNSDIV);
    Logger::printParam("ITERATIONS", ITERATIONS);
    Logger::printParam("CAMPARY_PRECISION", CAMPARY_PRECISION);
    Logger::printDash();
    rns_const_print(true);
    Logger::printDash();
    rns_eval_const_print();
    Logger::endSection(true);
    Logger::printSpace();
    run_test(ITERATIONS);
    Logger::endTestDescription();
    return 0;
}