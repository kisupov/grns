/*
 *  Scaling algorithms for the residue number system
 */

#ifndef GRNS_SCAL_CUH
#define GRNS_SCAL_CUH

#include "rnsbase.cuh"

#define RNS_P2_SCALING_FACTOR 1 << RNS_P2_SCALING_THRESHOLD // RNS power-of-two scaling factor

/*
 * For a given RNS number x = (x0,...,xn), this helper function computes r such that
 * X = sum( Mi * x_i * w_i (mod m_i) ) - r * M. Array s stores the computed values (xi * w_i) mod mi
 * where w_i is the modulo mi multiplicative inverse of Mi = M / mi.
 */
GCC_FORCEINLINE static int rns_rank_compute(int * x, int * s) {
    double fracl[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding down
    double fracu[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding up
    double suml = 0.0;
    double sumu = 0.0;
    int mrd[RNS_MODULI_SIZE];
    int mr = -1;

    //Computing ( (x_i * w_i) mod m_i ) / mi
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracl[i] = dmul_rd(s[i], RNS_MODULI_RECIP_RD[i]);
        fracu[i] = dmul_ru(s[i], RNS_MODULI_RECIP_RU[i]);
        //ddiv_rdu(&fracl[i], &fracu[i], (double)s[i], (double)RNS_MODULI[i]);
    }
    //Pairwise summation of the fractions
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
    //Discarding the fractional part
    auto whl = (unsigned int) suml; // Whole part
    auto whu = (unsigned int) sumu; // Whole part
    //Checking for quick return
    if(whl == whu) {
        return whl;
    } else {
        mrc(mrd, x); //Computing the mixed-radix representation of x
        mr = mrd[RNS_MODULI_SIZE - 1];
        return mr == 0 ? whu : whl;
    }
}

/*
 * For a given RNS number x = (x0,...,xn), which is guaranteed not to be too large,
 * this helper function computes r such that X = sum( Mi * x_i * w_i (mod m_i) ) - r * M.
 * Array s stores the computed values (xi * w_i) mod mi, where w_i is the modulo mi multiplicative inverse of Mi = M / mi.
 * This function performs faster than the previous one.
 */
GCC_FORCEINLINE static int rns_rank_compute_fast(int * s) {
    double fracu[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding up
    double sumu = 0.0;
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracu[i] = dmul_ru(s[i], RNS_MODULI_RECIP_RU[i]);
        //fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
    }
    sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
    return (int) sumu;
}

/*
 * This helper function performs one step of scaling by a power of two
 */
GCC_FORCEINLINE static void make_scaling_step(int *y, int k, unsigned int j, int pow2j, int * x, int * c) {
    long residue = 0; //X mod 2^j
    int multiple[RNS_MODULI_SIZE]; //X - (X mod pow2j)
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        //RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j-1][i] ->  M_i mod 2^j
        residue += (long)mod_mul(RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j - 1][i], c[i], pow2j);
    }
    //RNS_MODULI_PRODUCT_POW2_RESIDUES[j-1] ->  M mod 2^j
    long temp = (long)k * (long)RNS_MODULI_PRODUCT_POW2_RESIDUES[j - 1];
    residue = (residue - temp) % pow2j;
    if(residue < 0)
        residue += pow2j;
    for (int i = 0; i < RNS_MODULI_SIZE; i++) { // multiple[i] <- (remainder when X is divided by pow2j) mod m_i
        multiple[i] = residue % RNS_MODULI[i];
    }
    rns_sub(multiple, x, multiple); //multiple <- X - remainder
    //RNS_POW2_INVERSE[j-1][i] -> (2^j )^{-1} mod m_i
    rns_mul(y, multiple, RNS_POW2_INVERSE[j - 1]);
}

/*!
  * Scaling an RNS number by a power of 2: result = x / 2^D
  * @param result - pointer to the result (scaled number)
  * @param x - pointer to the RNS number to be scaled
  * @param D - exponent of the scaling factor
  */
GCC_FORCEINLINE void rns_scale2pow(int * result, int * x, unsigned int D) {
    rns_set(result, x); // result <- x
    int t = D / RNS_P2_SCALING_THRESHOLD;
    int k = 0;
    int c[RNS_MODULI_SIZE];
    //first step
    if (t > 0) {
        rns_mul(c, x, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = rns_rank_compute(x, c);
        make_scaling_step(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, x, c);
        t -= 1;
    }
    //second step
    while (t > 0) {
        rns_mul(c, result, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = rns_rank_compute_fast(c);
        make_scaling_step(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, result, c);
        t -= 1;
    }
    //third step
    unsigned int d = D % RNS_P2_SCALING_THRESHOLD;
    if (d > 0) {
        rns_mul(c, result, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = d < D ? rns_rank_compute_fast(c) : rns_rank_compute(result, c);
        make_scaling_step(result, k, d, 1 << d, result, c);
    }
}


/*
 * GPU functions
 */
namespace cuda{

    /********************* Single-threaded functions *********************/

    /*
     * For a given RNS number x = (x0,...,xn), this helper function computes r such that
     * X = sum( Mi * x_i * w_i (mod m_i) ) - r * M. Array s stores the computed values (xi * w_i) mod mi
     * where w_i is the modulo mi multiplicative inverse of Mi = M / mi.
     */
    DEVICE_CUDA_FORCEINLINE static int rns_rank_compute(int * x, int * s) {
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        int mrd[RNS_MODULI_SIZE];
        int mr = -1;
        //Computing ( (x_i * w_i) mod m_i ) / mi
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], cuda::RNS_MODULI_RECIP_RD[i]);
            fracu[i] = __dmul_ru(s[i], cuda::RNS_MODULI_RECIP_RU[i]);
        }
        //Pairwise summation of the fractions
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
        //Discarding the fractional part
        auto whl = (unsigned int) (suml);
        auto whu = (unsigned int) (sumu);
        //Checking for quick return
        if(whl == whu) {
            return whl;
        } else {
            cuda::mrc(mrd, x);
            mr = mrd[RNS_MODULI_SIZE - 1];
            return mr == 0 ? whu : whl;
        }
    }

    /*
     * For a given RNS number x = (x0,...,xn), which is guaranteed not to be too large,
     * this helper function computes r such that X = sum( Mi * x_i * w_i (mod m_i) ) - r * M.
     * Array s stores the computed values (xi * w_i) mod mi, where w_i is the modulo mi multiplicative inverse of Mi = M / mi.
     * This function performs faster than the previous one.
     */
    DEVICE_CUDA_FORCEINLINE static int rns_rank_compute_fast(int * x, int * s) {
        double fracu[RNS_MODULI_SIZE];
        double sumu = 0.0;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracu[i] = __dmul_ru(s[i], cuda::RNS_MODULI_RECIP_RU[i]);
        }
        sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
        return (int) sumu;
    }

    /*
     * This helper function performs one step of scaling by a power of two
     */
    DEVICE_CUDA_FORCEINLINE static void make_scaling_step(int * y, int k, unsigned int j, int pow2j, int * x, int * c) {
        long residue = 0; // X mod 2^j
        int multiple[RNS_MODULI_SIZE]; // X - (X mod pow2j)
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            //RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j-1][i] ->  M_i mod 2^j
            residue += (long)cuda::mod_mul(cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j - 1][i], c[i], pow2j);
        }
        //RNS_MODULI_PRODUCT_POW2_RESIDUES[j-1] ->  M mod 2^j
        long temp = (long)k * (long)cuda::RNS_MODULI_PRODUCT_POW2_RESIDUES[j - 1];
        residue = (residue - temp) % pow2j;
        if(residue < 0)
            residue += pow2j;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) { // multiple[i] <- (remainder when X is divided by pow2j) mod m_i
            multiple[i] = residue % cuda::RNS_MODULI[i];
        }
        cuda::rns_sub(multiple, x, multiple); //multiple <- X - remainder
        //RNS_POW2_INVERSE[j-1][i] -> (2^j )^{-1} mod m_i
        cuda::rns_mul(y, multiple, cuda::RNS_POW2_INVERSE[j - 1]);
    }

    /*!
      * Scaling an RNS number by a power of 2: result = x / 2^D
      * @param result - pointer to the result (scaled number)
      * @param x - pointer to the RNS number to be scaled
      * @param D - exponent of the scaling factor
      */
    DEVICE_CUDA_FORCEINLINE void rns_scale2pow(int *result, int * x, unsigned int D) {
        for (int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = x[i];
        }
        int t = D / RNS_P2_SCALING_THRESHOLD;
        int k = 0;
        int c[RNS_MODULI_SIZE];
        //first step
        if (t > 0) {
            cuda::rns_mul(c, x, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
            k = cuda::rns_rank_compute(x, c);
            cuda::make_scaling_step(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, x, c);
            t -= 1;
        }
        //second step
        while (t > 0) {
            cuda::rns_mul(c, result, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
            k = cuda::rns_rank_compute_fast(result, c);
            cuda::make_scaling_step(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, result, c);
            t -= 1;
        }
        //third step
        unsigned int d = D % RNS_P2_SCALING_THRESHOLD;
        if (d > 0) {
            cuda::rns_mul(c, result, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
            k = d < D ? cuda::rns_rank_compute(result, c) : cuda::rns_rank_compute_fast(result, c);
            cuda::make_scaling_step(result, k, d, 1 << d, result, c);
        }
    }

    /********************* Multi-threaded functions *********************/

    /*
     * For a given RNS number x = (x0,...,xn), this helper function computes k such that
     * X = sum( Mi * |xi * mult.inv(Mi)|_mi ) - k * M. Array c stores the computed values
     * |xi * mult.inv(Mi)|_mi, where mult.inv(Mi) is the modulo mi multiplicative inverse of Mi, i.e. M_i^{-1} mod mi
     * This function must be performed by n threads simultaneously within a single thread block.
     */
    /*DEVICE_CUDA_FORCEINLINE static void compute_k_thread(int * result, int * x, int c) {
        int residueId = threadIdx.x;
        int modulus = cuda::RNS_MODULI[residueId];
        __shared__ double s_upp[RNS_MODULI_SIZE];
        __shared__ double s_low[RNS_MODULI_SIZE];
        __shared__ bool return_flag;
        return_flag = false;
        int k_low, k_upp;
        s_low[residueId] = __ddiv_rd(c, (double) modulus);
        s_upp[residueId] = __ddiv_ru(c, (double) modulus);
        __syncthreads();
        for (unsigned int s = cuda::PRECEDING_POW2(RNS_MODULI_SIZE); s > 0; s >>= 1) {
            if (residueId < s && residueId + s < RNS_MODULI_SIZE) {
                s_low[residueId] = __dadd_rd((double) s_low[residueId], (double) s_low[residueId + s]);
                s_upp[residueId] = __dadd_ru((double) s_upp[residueId], (double) s_upp[residueId + s]);
            }
            __syncthreads();
        }
        k_low = (int) s_low[residueId];
        k_upp = (int) s_upp[residueId];
        if (residueId == 0) {
            if (k_low == k_upp) {
                *result = k_low;
                return_flag = true;
            }
        }
        __syncthreads();
        if (return_flag) {
            return;
        } else {
            if (residueId == 0) {
                int mr[RNS_MODULI_SIZE];
                cuda::mrc(mr, x); // parallel MRC should be used, see http://dx.doi.org/10.1109/ISCAS.2009.5117800
                if (mr[RNS_MODULI_SIZE - 1] == 0) {
                    *result = k_upp;
                } else{
                    *result = k_low;
                }
            }
        }
        __syncthreads();
    }*/

    /*
     * For a given RNS number x = (x0,...,xn), which is guaranteed not to be too large, this helper function computes k such that
     * X = sum( Mi * |xi * mult.inv(Mi)|_mi ) - k * M. Array c stores the computed values
     * |xi * mult.inv(Mi)|_mi, where mult.inv(Mi) is the modulo mi multiplicative inverse of Mi, i.e. M_i^{-1} mod mi
     * This function performs faster than the previous common function.
     * This function must be performed by n threads simultaneously within a single thread block.
     */
    /*DEVICE_CUDA_FORCEINLINE static void compute_k_fast_thread(int * result, int * x, int c){
        int residueId = threadIdx.x;
        __shared__ double S[RNS_MODULI_SIZE];
        S[residueId] = __ddiv_ru((double) c, (double) cuda::RNS_MODULI[residueId]);
        __syncthreads();
        for (unsigned int s = cuda::PRECEDING_POW2(RNS_MODULI_SIZE); s > 0; s >>= 1) {
            if (residueId < s && residueId + s < RNS_MODULI_SIZE) {
                S[residueId] = __dadd_ru(S[residueId], S[residueId + s]);
            }
            __syncthreads();
        }
        if (residueId == 0)
            *result = (int) S[0];
        __syncthreads();
    }*/

    /*
     * This helper function performs one step of scaling by a power of two
     * This function must be performed by n threads simultaneously within a single thread block.
     */
    /*DEVICE_CUDA_FORCEINLINE static void scale2powj_thread(int * y, int k, unsigned int j, int pow2j, int * x, int c) {
        int residueId = threadIdx.x;
        int modulus = cuda::RNS_MODULI[residueId];
        int multiple;
        __shared__ int residue[RNS_MODULI_SIZE]; // X mod 2^j
        //RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j-1][i] ->  M_i mod 2^j
        residue[residueId] = cuda::mod_mul(cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j - 1][residueId], c, pow2j);// (cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j - 1][residueId] * terms) % pow2j;
        __syncthreads();
        for (unsigned int s = cuda::PRECEDING_POW2(RNS_MODULI_SIZE); s > 0; s >>= 1) {
            if (residueId < s && residueId + s < RNS_MODULI_SIZE) {
                residue[residueId] = residue[residueId] + residue[residueId + s];
            }
            __syncthreads();
        }
        //RNS_MODULI_PRODUCT_POW2_RESIDUES[j-1] ->  M mod 2^j
        if(residueId == 0){
            residue[0] = (residue[0] - k * cuda::RNS_MODULI_PRODUCT_POW2_RESIDUES[j - 1]) % pow2j;
            if(residue[0] < 0){
                residue[0] += pow2j;
            }
        }
        __syncthreads();
        residue[residueId] = residue[0];
        multiple = residue[residueId] % modulus;
        multiple = x[residueId] - multiple;
        if (multiple < 0) {
            multiple += modulus;
        }
        //RNS_POW2_INVERSE[j-1][i] -> (2^j )^{-1} mod m_i
        y[residueId] = cuda::mod_mul(multiple, cuda::RNS_POW2_INVERSE[j - 1][residueId], modulus);   //( multiple * cuda::RNS_POW2_INVERSE[j - 1][residueId] ) % modulus;
    }*/

    /*!
     * Parallel (n threads) scaling an RNS number by a power of 2: result = x / 2^D.
     * This function must be performed by n threads simultaneously within a single thread block.
     * @param result - pointer to the result (scaled number)
     * @param x - pointer to the RNS number to be scaled
     * @param D - exponent of the scaling factor
     */
    /*DEVICE_CUDA_FORCEINLINE void rns_scale2pow_parallel(int * result, int * x, unsigned int D) {
        result[threadIdx.x] = x[threadIdx.x];
        __shared__ int k;
        int t;
        int c;
        int residueId = threadIdx.x;
        int modulus = cuda::RNS_MODULI[residueId];
        int inverse = cuda::RNS_PART_MODULI_PRODUCT_INVERSE[residueId];

        t = D / RNS_P2_SCALING_THRESHOLD;
        __syncthreads();
        //first step
        if (t > 0) {
            c = (x[residueId] * inverse) % modulus;
            cuda::compute_k_thread(&k, x, c);
            cuda::scale2powj_thread(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, x, c);
            t -= 1;
        }
        __syncthreads();
        //second step
        while (t > 0) {
            c = (result[residueId] * inverse) % modulus;
            cuda::compute_k_fast_thread(&k, result, c);
            cuda::scale2powj_thread(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, result, c);
            t -= 1;
            __syncthreads();
        }
        //third step
        unsigned int d = D % RNS_P2_SCALING_THRESHOLD;
        if (d > 0) {
            c =  cuda::mod_mul(result[residueId], inverse, modulus);
            if (d < D) {
                cuda::compute_k_fast_thread(&k, result, c);
            } else {
                cuda::compute_k_thread(&k, result, c);
            }
            cuda::scale2powj_thread(result, k, d, 1 << d, result, c);
            __syncthreads();
        }
    }*/

} //end of namespace


#endif  //GRNS_SCAL_CUH


