/*
 *  Test for measure the performance of the algorithms that calculate the RNS interval evaluation
 */

#include <stdio.h>
#include <iostream>
#include "../src/rnseval.cuh"
#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"

#define ITERATIONS 30000

/*
 *  Printing the error of the computed interval evaluation with respect
 *  to the exact relative value of an RNS number
 */
void printError(interval_ptr eval, er_float_ptr exact) {
    std::cout << "eval_low  = ";
    er_print(&eval->low);
    std::cout << "\t eval_upp  = ";
    er_print(&eval->upp);

    er_adjust(exact);
    if((er_cmp(&eval->low, exact) == 1) || (er_cmp(exact, &eval->upp) == 1)){
        std::cout << "\t error    = 100%. The RNS Interval Evaluation is wrong!\n";
    }
    else{
        er_float_ptr error = new er_float_t[1];
        er_sub(error, &eval->upp, &eval->low);
        er_div(error, error, exact);
        double derror;
        er_get_d(&derror, error);
        std::cout << "\t rel.error    = " << (derror) << std::endl;
        delete error;
    }
}

void resetResult(interval_t * res, int iterations){
    for(int i = 0; i < iterations; i++){
        er_set_d(&res[i].low, 0.0);
        er_set_d(&res[i].upp, 0.0);
    }
}

__global__ void resetResultCuda(interval_t * res, int iterations) {
    for(int i = 0; i < iterations; i++) {
        cuda::er_set_d(&res[i].low, 0.0);
        cuda::er_set_d(&res[i].upp, 0.0);
    }
}

/*
 * CUDA tests
 */

__global__ void run_rns_eval_compute(interval_t * res, int * x, int iterations){
    for (int i = 0; i < iterations; i++){
        cuda::rns_eval_compute(&res[i].low, &res[i].upp, &x[i * RNS_MODULI_SIZE]);
    }
}

__global__ void run_rns_eval_compute_fast(interval_t * res, int * x, int iterations){
    for (int i = 0; i < iterations; i++){
        cuda::rns_eval_compute_fast(&res[i].low, &res[i].upp, &x[i * RNS_MODULI_SIZE]);
    }
}

__global__ void run_rns_eval_compute_parallel(interval_t * res, int * x, int iterations){
    for (int i = 0; i < iterations; i++){
        cuda::rns_eval_compute_parallel(&res[i].low, &res[i].upp, &x[i * RNS_MODULI_SIZE]);
    }
}

// Main test
void run_test(int iterations) {
    Logger::printDash();
    InitCpuTimer();
    InitCudaTimer();

    // Host data
    int * hx = new int[iterations * RNS_MODULI_SIZE];
    interval_t * hresult = new interval_t[iterations];
    er_float_t * exact = new er_float_t[iterations];

    // Device data
    int * dx;
    interval_t * dresult;

    // Memory allocation
    cudaMalloc(&dx, sizeof(int) * iterations * RNS_MODULI_SIZE);
    cudaMalloc(&dresult, sizeof(interval_t) * iterations);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Generate inputs
    fill_random_array(hx, ITERATIONS, BND_RNS_MODULI_PRODUCT);

    //Copying data to the GPU
    cudaMemcpy(dx, hx, sizeof(int) * RNS_MODULI_SIZE * iterations, cudaMemcpyHostToDevice);


    // Computing exact results
    //---------------------------------------------------------
    for(int i = 0; i < iterations; i++){
        rns_fractional(&exact[i], &hx[i * RNS_MODULI_SIZE]);
    }
    std::cout << "exact = ";
    er_print(&exact[iterations - 1]);
    //---------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    PrintTimerName("[CPU] rns_eval_compute");
    resetResult(hresult, iterations);
    //Launch
    StartCpuTimer();
    for(int i = 0; i < iterations; i++){
        rns_eval_compute(&hresult[i].low, &hresult[i].upp, &hx[i * RNS_MODULI_SIZE]);
    }
    EndCpuTimer();
    PrintCpuTimer("took");
    printError(&hresult[iterations-1], &exact[iterations-1]);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CPU] rns_eval_compute_fast");
    resetResult(hresult, iterations);
    //Launch
    StartCpuTimer();
    for(int i = 0; i < iterations; i++){
        rns_eval_compute_fast(&hresult[i].low, &hresult[i].upp, &hx[i * RNS_MODULI_SIZE]);
    }
    EndCpuTimer();
    PrintCpuTimer("took");
    printError(&hresult[iterations-1], &exact[iterations-1]);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] rns_eval_compute");
    resetResult(hresult, iterations);
    resetResultCuda<<<1, 1>>>(dresult, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    run_rns_eval_compute<<<1,1>>>(dresult, dx, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    //Copying to the host
    cudaMemcpy(hresult, dresult, sizeof(interval_t) * iterations, cudaMemcpyDeviceToHost);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    printError(&hresult[iterations-1], &exact[iterations-1]);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] rns_eval_compute_fast");
    resetResult(hresult, iterations);
    resetResultCuda<<<1, 1>>>(dresult, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    run_rns_eval_compute_fast<<<1,1>>>(dresult, dx, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hresult, dresult, sizeof(interval_t) * iterations, cudaMemcpyDeviceToHost);
    printError(&hresult[iterations-1], &exact[iterations-1]);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] rns_eval_compute_parallel");
    resetResult(hresult, iterations);
    resetResultCuda<<<1, 1>>>(dresult, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    run_rns_eval_compute_parallel<<<1,RNS_MODULI_SIZE>>>(dresult, dx, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hresult, dresult, sizeof(interval_t) * iterations, cudaMemcpyDeviceToHost);
    printError(&hresult[iterations-1], &exact[iterations-1]);
    //---------------------------------------------------------

    // Cleanup
    delete [] hx;
    delete [] hresult;
    delete [] exact;
    cudaFree(dx);
    cudaFree(dresult);
}

int main() {
    cudaDeviceReset();
    rns_const_init();
    Logger::beginTestDescription(Logger::TEST_PERF_RNSEVAL);
    Logger::printParam("ITERATIONS", ITERATIONS);
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