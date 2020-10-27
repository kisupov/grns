/*
 *  Implementation of the floating-point interval evaluation for performing non-modular operations in RNS
 */

#ifndef GRNS_EVAL_CUH
#define GRNS_EVAL_CUH

#include "rnsbase.cuh"
#include "mrc.cuh"
#include <math.h>

/*
 * Printing the constants for the RNS interval evaluation
 */
void rns_eval_const_print() {
    printf("Constants of the RNS interval evaluation for %i moduli:\n", RNS_MODULI_SIZE);
    printf("- RNS_EVAL_RELATIVE_ERROR: %.10f\n", RNS_EVAL_RELATIVE_ERROR);
    printf("- RNS_EVAL_ACCURACY: %.17g\n", RNS_EVAL_ACCURACY);
    printf("- RNS_EVAL_REF_FACTOR: %i\n", RNS_EVAL_REF_FACTOR);
}

/*!
 * Computes the interval evaluation for a given RNS number
 * This is an improved version of the algorithm from IEEE Access paper (for reference, see README.md)
 * @param low - pointer to the lower bound of the result interval evaluation
 * @param upp - pointer to the upper bound of the result interval evaluation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void rns_eval_compute(er_float_ptr low, er_float_ptr upp, int * x) {
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
        //The improvement is that the refinement factor depends on the value of X
        int k = MAX(-(ceil(log2(sumu))+1), RNS_EVAL_REF_FACTOR);
        rns_mul(s, s, RNS_POW2[k]);
        for(int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
        }
        sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
        sumu -= (unsigned int) sumu;
        j += k;
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
    low->exp -= j;
    upp->exp -= j;
}


/*!
 * For a given RNS number, which is guaranteed not to be too large,
 * this function computes the interval evaluation faster than the previous common function.
 * @param low - pointer to the lower bound of the result interval evaluation
 * @param upp - pointer to the upper bound of the result interval evaluation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void rns_eval_compute_fast(er_float_ptr low, er_float_ptr upp, int * x) {
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
        //The improvement is that the refinement factor depends on the value of X
        int k = MAX(-(ceil(log2(sumu))+1), RNS_EVAL_REF_FACTOR);
        rns_mul(s, s, RNS_POW2[k]);
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
        }
        sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
        sumu -= (unsigned int) sumu;
        j += k;
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
    low->exp -= j;
    upp->exp -= j;
}


/*
 * GPU functions
 */
namespace cuda{

    /*!
     * Computes the interval evaluation for a given RNS number
     * This is an improved version of the algorithm from IEEE Access paper (for reference, see README.md)
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute(er_float_ptr low, er_float_ptr upp, int * x) {
        double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int  s[RNS_MODULI_SIZE];
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        cuda::rns_mul(s, x, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
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
            //The improvement is that the refinement factor depends on the value of X
            int k = MAX(-(ceil(log2(sumu))+1), cuda::RNS_EVAL_REF_FACTOR);
            cuda::rns_mul(s, s, cuda::RNS_POW2[k]);
            for(int i = 0; i < RNS_MODULI_SIZE; i++) {
                fracu[i] = __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]);
            }
            sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
            sumu = __dsub_ru(sumu, (unsigned int) sumu);
            j += k;
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
        low->exp -= j;
        upp->exp -= j;
    }


    /*!
     * For a given RNS number, which is guaranteed not to be too large,
     * this function computes the interval evaluation faster than the previous common function.
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_fast(er_float_ptr low, er_float_ptr upp, int * x) {
        double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int s[RNS_MODULI_SIZE];
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        cuda::rns_mul(s, x, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
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
            //The improvement is that the refinement factor depends on the value of X
            int k = MAX(-(ceil(log2(sumu))+1), cuda::RNS_EVAL_REF_FACTOR);
            cuda::rns_mul(s, s, cuda::RNS_POW2[k]);
            for(int i = 0; i < RNS_MODULI_SIZE; i++) {
                fracu[i] = __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]);
            }
            sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
            sumu = __dsub_ru(sumu, (unsigned int) sumu);
            j += k;
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
        low->exp -= j;
        upp->exp -= j;
    }

    /*!
     * Computes the interval evaluation for a given RNS number in parallel
     * This routine must be executed by RNS_MODULI_SIZE threads concurrently.
     * This is an improved version of the algorithm from IEEE Access paper (for reference, see README.md)
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_parallel(er_float_ptr low, er_float_ptr upp, int * x) {
        double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int modulus = cuda::RNS_MODULI[threadIdx.x];
        int mr;
        __shared__ double fracl[RNS_MODULI_SIZE];
        __shared__ double fracu[RNS_MODULI_SIZE];
        __shared__ bool control;
        __shared__ bool ambiguity;

        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        int s = cuda::mod_mul(x[threadIdx.x], cuda::RNS_PART_MODULI_PRODUCT_INVERSE[threadIdx.x], modulus, cuda::RNS_MODULI_RECIPROCAL[threadIdx.x]);
        fracl[threadIdx.x] = __ddiv_rd(s, (double) modulus);
        fracu[threadIdx.x] = __ddiv_ru(s, (double) modulus);
        __syncthreads();

        //Parallel reduction. The result in fracl[0] and fracu[0]
        for (unsigned int i = RNS_PARALLEL_REDUCTION_IDX; i > 0; i >>= 1) {
            if (threadIdx.x < i && threadIdx.x + i < RNS_MODULI_SIZE) {
                fracl[threadIdx.x] = __dadd_rd(fracl[threadIdx.x], fracl[threadIdx.x + i]);
                fracu[threadIdx.x] = __dadd_ru(fracu[threadIdx.x], fracu[threadIdx.x + i]);
            }
            __syncthreads();
        }

        //Check for zero and ambiguity
        if (threadIdx.x == 0) {
            //Check for zero
            control = false;
            if(fracl[0] == 0 && fracu[0] == 0){
                cuda::er_set(low, &cuda::RNS_EVAL_ZERO_BOUND);
                cuda::er_set(upp, &cuda::RNS_EVAL_ZERO_BOUND);
                control = true;
            }
            //Splitting into whole and fractional parts
            unsigned int whl = (unsigned int) (fracl[0]);
            unsigned int whu = (unsigned int) (fracu[0]);
            fracl[0] = __dsub_rd(fracl[0], whl);    // lower bound
            fracu[0] = __dsub_ru(fracu[0], whu);    // upper bound

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
        //__syncthreads();

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
            if (huge || fracu[0] >= accuracy_constant) { // Refinement is not required
                if (!tiny)  cuda::er_set_d(low, fracl[0]);
                if (!huge)  cuda::er_set_d(upp, fracu[0]);
            } else{
                control = true;
            }
        }
        __syncthreads();

        /*
         * Incremental refinement
         */
        if(control){
            int j = 0;
            while (fracu[0] < accuracy_constant) {
                //The improvement is that the refinement factor depends on the value of X
                int k = MAX(-(ceil(log2(fracu[0]))+1), cuda::RNS_EVAL_REF_FACTOR);
                s = cuda::mod_mul(s, cuda::RNS_POW2[k][threadIdx.x], modulus, cuda::RNS_MODULI_RECIPROCAL[threadIdx.x]);
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
                j += k;
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
                low->exp -= j;
                upp->exp -= j;
            }
        }
    }

} //end of namespace

#endif  //GRNS_EVAL_CUH


