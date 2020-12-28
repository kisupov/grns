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
    int mrd[RNS_MODULI_SIZE];
    int mr = -1;
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
    auto whl = (unsigned int) suml; // Whole part
    auto whu = (unsigned int) sumu; // Whole part
    suml = suml - whl;    // Fractional part
    sumu = sumu - whu;    // Fractional part
    //Assign the computed values to the result
    er_set_d(low, suml);
    er_set_d(upp, sumu);
    //Check for ambiguity
    if(whl != whu) {
        perform_mrc(mrd, x); //Computing the mixed-radix representation of x
        mr = mrd[RNS_MODULI_SIZE - 1];
    }
    //Adjust if ambiguity was found
    if(mr > 0){
        er_set(upp, &RNS_EVAL_INV_UNIT.upp);
        return;
    }
    if(mr == 0){
        er_set(low, &RNS_EVAL_UNIT.low);
    }
    // Refinement is not required
    if(sumu >= RNS_EVAL_ACCURACY){
        return;
    }
    //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
    int K = 0;
    while (sumu < RNS_EVAL_ACCURACY) {
        //The improvement is that the refinement factor depends on the value of X
        int k = MAX(-(ceil(log2(sumu))+1), RNS_EVAL_REF_FACTOR);
        rns_mul(s, s, RNS_POW2[k]);
        for(int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
        }
        sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
        sumu -= (unsigned int) sumu;
        K += k;
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
    int K = 0;
    while (sumu < RNS_EVAL_ACCURACY) {
        //The improvement is that the refinement factor depends on the value of X
        int k = MAX(-(ceil(log2(sumu))+1), RNS_EVAL_REF_FACTOR);
        rns_mul(s, s, RNS_POW2[k]);
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
        }
        sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
        sumu -= (unsigned int) sumu;
        K += k;
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
    low->exp -= K;
    upp->exp -= K;
}


/*
 * GPU functions
 */
namespace cuda{

    /*!
     * Computes the interval evaluation for a given RNS number
     * This is an improved version of an algorithm from the IEEE Access paper (for reference, see README.md)
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute(er_float_ptr low, er_float_ptr upp, int * x) {
        constexpr double moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        const double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int  s[RNS_MODULI_SIZE];
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        int mrd[RNS_MODULI_SIZE];
        int mr = -1;
        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        //1.0 / moduli[i] is evaluated at compile time
        cuda::rns_mul(s, x, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], 1.0 / moduli[i]);
            fracu[i] = __dmul_ru(s[i], 1.0 / moduli[i]);
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
        auto whl = (unsigned int) (suml);
        auto whu = (unsigned int) (sumu);
        suml = __dsub_rd(suml, whl);    // lower bound
        sumu = __dsub_ru(sumu, whu);    // upper bound
        //Assign the computed values to the result
        cuda::er_set_d(low, suml);
        cuda::er_set_d(upp, sumu);
        //Check for ambiguity
        if(whl != whu) {
            cuda::perform_mrc(mrd, x); //Computing the mixed-radix representation of x
            mr = mrd[RNS_MODULI_SIZE - 1];
        }
        //Adjust if ambiguity was found
        if(mr > 0){
            cuda::er_set(upp, &cuda::RNS_EVAL_INV_UNIT.upp);
            return;
        }
        if(mr == 0){
            cuda::er_set(low, &cuda::RNS_EVAL_UNIT.low);
        }
        // Refinement is not required
        if(sumu >= accuracy_constant){
            return;
        }
        //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
        int K = 0;
        while (sumu < accuracy_constant) {
            //The improvement is that the refinement factor depends on the value of X
            int k = MAX(-(ceil(log2(sumu))+1), cuda::RNS_EVAL_REF_FACTOR);
            cuda::rns_mul(s, s, cuda::RNS_POW2[k]);
            for(int i = 0; i < RNS_MODULI_SIZE; i++) {
                fracu[i] = __dmul_ru(s[i], 1.0 / moduli[i]);
            }
            sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
            sumu = __dsub_ru(sumu, (unsigned int) sumu);
            K += k;
        }
        // Computing the shifted lower bound
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], 1.0 / moduli[i]);
        }
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        suml = __dsub_rd(suml, (unsigned int) suml);
        //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
        cuda::er_set_d(low, suml);
        cuda::er_set_d(upp, sumu);
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
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_fast(er_float_ptr low, er_float_ptr upp, int * x) {
        constexpr double moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        const double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int s[RNS_MODULI_SIZE];
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        //1.0 / moduli[i] is evaluated at compile time
        cuda::rns_mul(s, x, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], 1.0 / moduli[i]);
            fracu[i] = __dmul_ru(s[i], 1.0 / moduli[i]);
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
        int K = 0;
        while (sumu < accuracy_constant) {
            //The improvement is that the refinement factor depends on the value of X
            int k = MAX(-(ceil(log2(sumu))+1), cuda::RNS_EVAL_REF_FACTOR);
            cuda::rns_mul(s, s, cuda::RNS_POW2[k]);
            for(int i = 0; i < RNS_MODULI_SIZE; i++) {
                fracu[i] = __dmul_ru(s[i], 1.0 / moduli[i]);
            }
            sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
            sumu = __dsub_ru(sumu, (unsigned int) sumu);
            K += k;
        }
        // Computing the shifted lower bound
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], 1.0 / moduli[i]);
        }
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        suml = __dsub_rd(suml, (unsigned int) suml);
        //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
        cuda::er_set_d(low, suml);
        cuda::er_set_d(upp, sumu);
        low->exp -= K;
        upp->exp -= K;
    }

    /*!
     * Computes the interval evaluation for a given RNS number in parallel
     * This routine must be executed by RNS_MODULI_SIZE threads concurrently.
     * This is an improved version of an algorithm from the IEEE Access paper (for reference, see README.md)
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_parallel(er_float_ptr low, er_float_ptr upp, int * x) {
        const double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        const int modulus = cuda::RNS_MODULI[threadIdx.x];
        int mr = -1;
        __shared__ double shl;
        __shared__ double shu;
        __shared__ int shm;

        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        int s = cuda::mod_mul(x[threadIdx.x], cuda::RNS_PART_MODULI_PRODUCT_INVERSE[threadIdx.x], modulus);
        //Reduction
        double suml = cuda::block_reduce_sum_rd(__dmul_rd(s, 1.0 / modulus), RNS_MODULI_SIZE);
        double sumu = cuda::block_reduce_sum_ru(__dmul_ru(s, 1.0 / modulus), RNS_MODULI_SIZE);
        //Broadcast sums among all threads
        if(threadIdx.x == 0){
            shl = suml;
            shu = sumu;
        }
        __syncthreads();
        suml = shl;
        sumu = shu;
        //Check for zero
        if(suml == 0 && sumu == 0){
            if(threadIdx.x == 0){
                cuda::er_set(low, &cuda::RNS_EVAL_ZERO_BOUND);
                cuda::er_set(upp, &cuda::RNS_EVAL_ZERO_BOUND);
            }
            return;
        }
        //Splitting into whole and fractional parts
        auto whl = (unsigned int) (suml);
        auto whu = (unsigned int) (sumu);
        suml = __dsub_rd(suml, whl);    // lower bound
        sumu = __dsub_ru(sumu, whu);    // upper bound
        //Assign the computed values to the result
        if(threadIdx.x == 0){
            cuda::er_set_d(low, suml);
            cuda::er_set_d(upp, sumu);
        }
        //Check for ambiguity
        if(whl != whu) {
            mr = cuda::get_mrmsd_parallel(x); // The result is stored in mr (only for thread 0)
            //Broadcast mr among all threads
            if (threadIdx.x == 0) shm = mr;
            __syncthreads();
            mr = shm;
        }
        //Adjust if ambiguity was found
        if(mr > 0){
            if(threadIdx.x == 0) cuda::er_set(upp, &cuda::RNS_EVAL_INV_UNIT.upp);
            return;
        }
        if(mr == 0){
            if(threadIdx.x == 0) cuda::er_set(low, &cuda::RNS_EVAL_UNIT.low);
        }
        // Refinement is not required
        if(sumu >= accuracy_constant){
            return;
        }

        //Need more accuracy. Performing a refinement loop with stepwise calculation of the upper bound
        int K = 0;
        while (sumu < accuracy_constant) {
            //The improvement is that the refinement factor depends on the value of X
            int k = MAX(-(ceil(log2(sumu))+1), cuda::RNS_EVAL_REF_FACTOR);
            s = cuda::mod_mul(s, cuda::RNS_POW2[k][threadIdx.x], modulus);
            sumu = cuda::block_reduce_sum_ru(__dmul_ru(s, 1.0 / modulus), RNS_MODULI_SIZE);
            //Broadcast sums among all threads
            if(threadIdx.x == 0) shu = sumu;
            __syncthreads();
            sumu = shu;
            sumu = __dsub_ru(sumu, (unsigned int)sumu); // upper bound
            K += k;
        }
        //Computing the lower bound, broadcast does not required
        suml = cuda::block_reduce_sum_rd(__dmul_rd(s, 1.0 / modulus), RNS_MODULI_SIZE);
        suml = __dsub_rd(suml, (unsigned int)suml);
        if(threadIdx.x == 0){
            //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
            cuda::er_set_d(low, suml);
            cuda::er_set_d(upp, sumu);
            low->exp -= K;
            upp->exp -= K;
        }
    }

} //end of namespace

#endif  //GRNS_EVAL_CUH


