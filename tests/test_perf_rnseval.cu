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


/////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of the original algorithm for computing the floating point interval evaluation
// from IEEE Access paper (for reference, see README.md)
// Currently, this implementation is used only for test
/////////////////////////////////////////////////////////////////////////////////////////////

/*!
 * Computes the interval evaluation for a given RNS number
 * @param low - pointer to the lower bound of the result interval evaluation
 * @param upp - pointer to the upper bound of the result interval evaluation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void rns_eval_compute_origin(er_float_ptr low, er_float_ptr upp, int * x) {
    int s[RNS_MODULI_SIZE]; //Array of x_i * w_i (mod m_i)
    double fracl[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding down
    double fracu[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding up
    double suml = 0.0; //Rounded downward sum
    double sumu = 0.0; //Rounded upward sum
    //Checking for zero
    if(rns_check_zero(x)){
        er_set(low, &RNS_EVAL_ZERO_BOUND);
        er_set(upp, &RNS_EVAL_ZERO_BOUND);
        return;
    }
    //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
    rns_mul(s, x, RNS_PART_MODULI_PRODUCT_INVERSE);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        ddiv_rdu(&fracl[i], &fracu[i], (double)s[i], (double)RNS_MODULI[i]);
    }
    //Pairwise summation of the fractions
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
    //Splitting into whole and fractional parts
    unsigned int whl = (unsigned int) suml; // Whole part
    unsigned int whu = (unsigned int) sumu; // Whole part
    suml = suml - whl;    // Fractional part
    sumu = sumu - whu;    // Fractional part
    //Checking the correctness and adjusting
    bool huge = false;
    bool tiny = false;
    if (whl != whu) { //Interval evaluation is wrong
        int mr[RNS_MODULI_SIZE];
        perform_mrc(mr, x); //Computing the mixed-radix representation of x
        if (mr[RNS_MODULI_SIZE - 1] == 0) {
            tiny = true; //Number is too small, the lower bound is incorrect
            er_set(low, &RNS_EVAL_UNIT.low);
        } else {
            huge = true; //Number is too large, the upper bound is incorrect
            er_set(upp, &RNS_EVAL_INV_UNIT.upp);
        }
    }
    /*
     * Accuracy checking
     * If the lower bound is incorrectly calculated (the number is too small), then refinement may be required;
     * If the upper bound is incorrectly calculated (the number is too large), no refinement is required.
    */
    if (huge || sumu >= RNS_EVAL_ACCURACY) { //Refinement is not required
        if (!tiny)  er_set_d(low, suml);
        if (!huge)  er_set_d(upp, sumu);
        return;
    }
    //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
    int j = 0;
    while (sumu < RNS_EVAL_ACCURACY) {
        rns_mul(s, s, RNS_POW2[RNS_EVAL_REF_FACTOR]);
        for(int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
        }
        sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
        sumu -= (unsigned int) sumu;
        j = j + 1;
    }
    //Computing the shifted lower bound
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracl[i] = ddiv_rd((double)s[i], (double)RNS_MODULI[i]);
    }
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    suml -= (unsigned int) suml;
    //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
    er_set_d(low, suml);
    er_set_d(upp, sumu);
    int K = RNS_EVAL_REF_FACTOR * j;
    low->exp -= K;
    upp->exp -= K;
}

/*!
 * For a given RNS number, which is guaranteed not to be too large,
 * this function computes the interval evaluation faster than the previous common function.
 * @param low - pointer to the lower bound of the result interval evaluation
 * @param upp - pointer to the upper bound of the result interval evaluation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void rns_eval_compute_fast_origin(er_float_ptr low, er_float_ptr upp, int * x) {
    int s[RNS_MODULI_SIZE];
    double fracl[RNS_MODULI_SIZE];
    double fracu[RNS_MODULI_SIZE];
    double suml = 0.0;
    double sumu = 0.0;
    //Checking for zero
    if(rns_check_zero(x)){
        er_set(low, &RNS_EVAL_ZERO_BOUND);
        er_set(upp, &RNS_EVAL_ZERO_BOUND);
        return;
    }
    //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
    rns_mul(s, x, RNS_PART_MODULI_PRODUCT_INVERSE);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        ddiv_rdu(&fracl[i], &fracu[i], (double)s[i], (double)RNS_MODULI[i]);
    }
    //Pairwise summation of the fractions
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
    //Dropping integer parts
    suml -= (unsigned int) suml;  //Lower bound
    sumu -= (unsigned int) sumu;  //Upper bound
    //Accuracy checking
    if (sumu >= RNS_EVAL_ACCURACY) {
        er_set_d(low, suml);
        er_set_d(upp, sumu);
        return;
    }
    //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
    int j = 0;
    while (sumu < RNS_EVAL_ACCURACY) {
        rns_mul(s, s, RNS_POW2[RNS_EVAL_REF_FACTOR]);
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
        }
        sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
        sumu -= (unsigned int) sumu;
        j = j + 1;
    }
    //Computing the shifted lower bound
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracl[i] = ddiv_rd((double)s[i], (double)RNS_MODULI[i]);
    }
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    suml -= (unsigned int) suml;
    //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
    er_set_d(low, suml);
    er_set_d(upp, sumu);
    int K = RNS_EVAL_REF_FACTOR * j;
    low->exp -= K;
    upp->exp -= K;
}

namespace cuda{

    /*!
     * Computes the interval evaluation for a given RNS number
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_origin(er_float_ptr low, er_float_ptr upp, int * x) {
        double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int  s[RNS_MODULI_SIZE];
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            s[i] = cuda::mod_mul(x[i], cuda::RNS_PART_MODULI_PRODUCT_INVERSE[i], cuda::RNS_MODULI[i]);
            fracl[i] = __ddiv_rd(s[i], (double) cuda::RNS_MODULI[i]);
            fracu[i] = __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]);
        }
        //Pairwise summation of the fractions
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
        //Checking for zero
        if (suml == 0 && sumu == 0) {
            cuda::er_set(low, &cuda::RNS_EVAL_ZERO_BOUND);
            cuda::er_set(upp, &cuda::RNS_EVAL_ZERO_BOUND);
            return;
        }
        //Splitting into whole and fractional parts
        unsigned int whl = (unsigned int) (suml);
        unsigned int whu = (unsigned int) (sumu);
        suml = __dsub_rd(suml, whl);    // lower bound
        sumu = __dsub_ru(sumu, whu);    // upper bound
        //Checking the correctness and adjusting
        bool huge = false;
        bool tiny = false;
        if (whl != whu) { //Interval evaluation is wrong
            int mr[RNS_MODULI_SIZE];
            cuda::perform_mrc(mr, x); //Computing the mixed-radix representation of x
            if (mr[RNS_MODULI_SIZE - 1] == 0) {
                tiny = true; //Number is too small, the lower bound is incorrect
                cuda::er_set(low, &cuda::RNS_EVAL_UNIT.low);
            } else {
                huge = true;  // Number is too large, incorrect upper bound
                cuda::er_set(upp, &cuda::RNS_EVAL_INV_UNIT.upp);
            }
        }
        /*
         * Accuracy checking
         * If the lower bound is incorrectly calculated (the number is too small), then refinement may be required;
         * If the upper bound is incorrectly calculated (the number is too large), no refinement is required.
        */
        if (huge || sumu >= accuracy_constant) { // Refinement is not required
            if (!tiny)  cuda::er_set_d(low, suml);
            if (!huge)  cuda::er_set_d(upp, sumu);
            return;
        }
        //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
        int j = 0;
        while (sumu < accuracy_constant) {
            for(int i = 0; i < RNS_MODULI_SIZE; i++) {
                s[i] = cuda::mod_mul(s[i], cuda::RNS_POW2[cuda::RNS_EVAL_REF_FACTOR][i], cuda::RNS_MODULI[i]);
                fracu[i] = __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]);
            }
            sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
            sumu = __dsub_ru(sumu, (unsigned int) sumu);
            j = j + 1;
        }
        // Computing the shifted lower bound
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __ddiv_rd(s[i], (double) cuda::RNS_MODULI[i]);
        }
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        suml = __dsub_rd(suml, (unsigned int) suml);
        //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
        cuda::er_set_d(low, suml);
        cuda::er_set_d(upp, sumu);
        int K = cuda::RNS_EVAL_REF_FACTOR * j;
        low->exp -= K;
        upp->exp -= K;
    }

    /*!
     * For a given RNS number, which is guaranteed not to be too large,
     * this function computes the interval evaluation faster than the previous common function.
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_fast_origin(er_float_ptr low, er_float_ptr upp, int * x) {
        double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int s[RNS_MODULI_SIZE];
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            s[i] = cuda::mod_mul(x[i], cuda::RNS_PART_MODULI_PRODUCT_INVERSE[i], cuda::RNS_MODULI[i]);
            fracl[i] = __ddiv_rd(s[i], (double) cuda::RNS_MODULI[i]);
            fracu[i] = __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]);
        }
        //Pairwise summation of the fractions
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
        //Checking for zero
        if (suml == 0 && sumu == 0) {
            cuda::er_set(low, &cuda::RNS_EVAL_ZERO_BOUND);
            cuda::er_set(upp, &cuda::RNS_EVAL_ZERO_BOUND);
            return;
        }
        //Dropping integer parts
        suml = __dsub_rd(suml, (unsigned int) suml); //Lower bound
        sumu = __dsub_ru(sumu, (unsigned int) sumu); //Upper bound
        //Accuracy checking
        if (sumu >= accuracy_constant) {
            cuda::er_set_d(low, suml);
            cuda::er_set_d(upp, sumu);
            return;
        }
        //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
        int j = 0;
        while (sumu < accuracy_constant) {
            for(int i = 0; i < RNS_MODULI_SIZE; i++) {
                s[i] = cuda::mod_mul(s[i], cuda::RNS_POW2[cuda::RNS_EVAL_REF_FACTOR][i], cuda::RNS_MODULI[i]);
                fracu[i] = __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]);
            }
            sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
            sumu = __dsub_ru(sumu, (unsigned int) sumu);
            j = j + 1;
        }
        // Computing the shifted lower bound
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __ddiv_rd(s[i], (double) cuda::RNS_MODULI[i]);
        }
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        suml = __dsub_rd(suml, (unsigned int) suml);
        //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
        cuda::er_set_d(low, suml);
        cuda::er_set_d(upp, sumu);
        int K = cuda::RNS_EVAL_REF_FACTOR * j;
        low->exp -= K;
        upp->exp -= K;
    }


    /*!
     * Computes the interval evaluation for a given RNS number in parallel
     * This routine must be executed by RNS_MODULI_SIZE threads concurrently.
     * This is the original version of the algorithm from IEEE Access paper (for reference, see README.md)
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_parallel_origin(er_float_ptr low, er_float_ptr upp, int * x) {
        double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int modulus = cuda::RNS_MODULI[threadIdx.x];
        double suml;
        double sumu;
        int mr;
        __shared__ double fracl[RNS_MODULI_SIZE];
        __shared__ double fracu[RNS_MODULI_SIZE];
        __shared__ bool control;
        __shared__ bool ambiguity;

        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        int s = cuda::mod_mul(x[threadIdx.x], cuda::RNS_PART_MODULI_PRODUCT_INVERSE[threadIdx.x], modulus);
        fracl[threadIdx.x] = __ddiv_rd(s, (double) modulus);
        fracu[threadIdx.x] = __ddiv_ru(s, (double) modulus);
        __syncthreads();

        //Parallel reduction
        for (unsigned int i = RNS_PARALLEL_REDUCTION_IDX; i > 0; i >>= 1) {
            if (threadIdx.x < i && threadIdx.x + i < RNS_MODULI_SIZE) {
                fracl[threadIdx.x] = __dadd_rd(fracl[threadIdx.x], fracl[threadIdx.x + i]);
                fracu[threadIdx.x] = __dadd_ru(fracu[threadIdx.x], fracu[threadIdx.x + i]);
            }
            __syncthreads();
        }

        //Check for zero and ambiguity
        if (threadIdx.x == 0) {
            //Splitting into whole and fractional parts
            unsigned int whl = (unsigned int) (fracl[0]);
            unsigned int whu = (unsigned int) (fracu[0]);
            suml = __dsub_rd(fracl[0], whl);    // lower bound
            sumu = __dsub_ru(fracu[0], whu);    // upper bound
            //Check for zero
            control = false;
            if(fracl[threadIdx.x] == 0 && fracu[threadIdx.x] == 0){
                cuda::er_set(low, &cuda::RNS_EVAL_ZERO_BOUND);
                cuda::er_set(upp, &cuda::RNS_EVAL_ZERO_BOUND);
                control = true;
            }
            //Check for ambiguity
            ambiguity = whl != whu;
        }
        __syncthreads();
        //Number is zero
        if( control ){ return; }
        //Ambiguity case, perform mixed-radix conversion
        if(ambiguity){
            mr = cuda::get_mrmsd_parallel(x); // The result is stored in mr (only for thread 0)
        }

        //Resolve ambiguity and check accuracy
        if(threadIdx.x == 0){
            bool huge = false;
            bool tiny = false;
            if(ambiguity){
                if (mr == 0) {
                    tiny = true; //Number is too small, the lower bound is incorrect
                    cuda::er_set(low, &cuda::RNS_EVAL_UNIT.low);
                } else {
                    huge = true;  // Number is too large, incorrect upper bound
                    cuda::er_set(upp, &cuda::RNS_EVAL_INV_UNIT.upp);
                }
            }
            /*
            * Accuracy checking
            * If the lower bound is incorrectly calculated (the number is too small), then refinement may be required;
            * If the upper bound is incorrectly calculated (the number is too large), no refinement is required.
            */
            if (huge || sumu >= accuracy_constant) { // Refinement is not required
                if (!tiny)  cuda::er_set_d(low, suml);
                if (!huge)  cuda::er_set_d(upp, sumu);
            } else{
                control = true;
                fracu[0] = 0;
            }
        }
        __syncthreads();

        /*
         * Incremental refinement
         */
        if(control){
            int j = 0;
            while (fracu[0] < accuracy_constant) {
                s = cuda::mod_mul(s, cuda::RNS_POW2[cuda::RNS_EVAL_REF_FACTOR][threadIdx.x], modulus);
                fracu[threadIdx.x] = __ddiv_ru(s, (double) modulus);
                __syncthreads();
                //Parallel reduction
                for (unsigned int i = RNS_PARALLEL_REDUCTION_IDX; i > 0; i >>= 1) {
                    if (threadIdx.x < i && threadIdx.x + i < RNS_MODULI_SIZE) {
                        fracu[threadIdx.x] = __dadd_ru(fracu[threadIdx.x], fracu[threadIdx.x + i]);
                    }
                    __syncthreads();
                }
                if(threadIdx.x == 0){
                    fracu[0] = __dsub_ru(fracu[0], (unsigned int)fracu[0]);    // upper bound
                }
                j = j + 1;
                __syncthreads();
            }

            /*
             * Computing the lower bound
             */
            fracl[threadIdx.x] = __ddiv_rd(s, (double) modulus);
            __syncthreads();
            for (unsigned int i = RNS_PARALLEL_REDUCTION_IDX; i > 0; i >>= 1) {
                if (threadIdx.x < i && threadIdx.x + i < RNS_MODULI_SIZE) {
                    fracl[threadIdx.x] = __dadd_rd(fracl[threadIdx.x], fracl[threadIdx.x + i]);
                }
                __syncthreads();
            }
            if(threadIdx.x == 0){
                fracl[0] = __dsub_rd(fracl[0], (unsigned int)fracl[0]);    // upper bound
                //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
                cuda::er_set_d(low, fracl[0]);
                cuda::er_set_d(upp, fracu[0]);
                int K = cuda::RNS_EVAL_REF_FACTOR * j;
                low->exp -= K;
                upp->exp -= K;
            }
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of the tests
/////////////////////////////////////////////////////////////////////////////////////////////

/*
 *  Printing the error of the computed interval evaluation with respect
 *  to the exact relative value of an RNS number
 */
void printError(interval_ptr eval, er_float_ptr exact) {
    std::cout << "eval_low  = ";
    er_print(&eval->low);
    std::cout << "\neval_upp  = ";
    er_print(&eval->upp);

    er_adjust(exact);
    if((er_cmp(&eval->low, exact) == 1) || (er_cmp(exact, &eval->upp) == 1)){
        std::cout << "\nerror    = 100%. The RNS Interval Evaluation is wrong!\n";
    }
    else{
        er_float_ptr error = new er_float_t[1];
        er_sub(error, &eval->upp, &eval->low);
        er_div(error, error, exact);
        double derror;
        er_get_d(&derror, error);
        std::cout << "\nrel.error    = " << (derror) << std::endl;
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