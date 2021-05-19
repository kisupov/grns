/*
 *  Mixed-radix conversion
 */

#ifndef GRNS_MRC_CUH
#define GRNS_MRC_CUH

#include "rnsbase.cuh"

/*
 * Pairwise comparison of the mixed-radix digits
 */
GCC_FORCEINLINE static int mrs_cmp(int * x, int * y) {
    for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
        if (x[i] > y[i]) {
            return 1;
        } else if (y[i] > x[i]) {
            return -1;
        }
    }
    return 0;
}

/*!
 * Computes the mixed-radix representation for a given RNS number using the Szabo and Tanaka's algorithm
 * for details, see N. S. Szabo and R. I. Tanaka, Residue Arithmetic and Its Application to Computer Technology
 * (McGraw-Hill, New York, 1967).
 * @param mr - pointer to the result mixed-radix representation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void mrc(int * mr, int * x) {
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mr[i] = x[i];
        for (int j = 0; j < i; j++) {
            if (mr[i] < mr[j]) {
                long tmp = (long)RNS_MODULI[i] - (long)mr[j] + (long)mr[i];
                mr[i] = (int)tmp;
            } else {
                mr[i] = mr[i] - mr[j];
            }
            mr[i] = mod_mul(mr[i], MRC_MULT_INV[j][i], RNS_MODULI[i]);
        }
    }
}

/*!
 * Converts x from mixed-radix system to binary system.
 * The result is stored in target
 */
GCC_FORCEINLINE void mrs_to_binary(mpz_t target, int * x) {
    mpz_t term;
    mpz_init(term);
    mpz_set_ui(target, 0);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_ui(term, MRS_BASES[i], x[i]);
        mpz_add(target, target, term);
    }
    mpz_clear(term);
}


/*!
 * Compares RNS numbers using mixed-radix conversion
 * @return 1, if x > y; -1, if x < y; 0, if x = y
 */
GCC_FORCEINLINE int mrc_compare_rns(int * x, int * y) {
    int mx[RNS_MODULI_SIZE];
    int my[RNS_MODULI_SIZE];
    mrc(mx, x);
    mrc(my, y);
    return mrs_cmp(mx, my);
}


/*
 * GPU functions
 */
namespace cuda{

    /*
     * Pairwise comparison of the mixed-radix digits
     */
    DEVICE_CUDA_FORCEINLINE static int mrs_cmp(int * x, int * y) {
        for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
            if (x[i] > y[i]) {
                return 1;
            } else if (y[i] > x[i]) {
                return -1;
            }
        }
        return 0;
    }

    /*!
     * Computes the mixed-radix representation for a given RNS number using the Szabo and Tanaka's algorithm
     * for details, see N. S. Szabo and R. I. Tanaka, Residue Arithmetic and Its Application to Computer Technology
     * (McGraw-Hill, New York, 1967).
     * @param mr - pointer to the result mixed-radix representation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void mrc(int * mr, int * x) {
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            mr[i] = x[i];
            for (int j = 0; j < i; j++) {
                if (mr[i] < mr[j]) {
                    mr[i] = cuda::mod_psub(mr[i], mr[j], cuda::RNS_MODULI[i]);
                } else {
                    mr[i] = mr[i] - mr[j];
                }
                mr[i] = cuda::mod_mul(mr[i], cuda::MRC_MULT_INV[j][i], cuda::RNS_MODULI[i]);
            }
        }
    }

    /*!
     * Parallel (pipeline) computation of the mixed-radix representation of an RNS number
     * This routine must be executed by RNS_MODULI_SIZE threads concurrently.
     * @param mr - pointer to the result mixed-radix representation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void mrc_pipeline(int * mr, int * x){
        __shared__ int v[RNS_MODULI_SIZE];
        int k = threadIdx.x;
        int m = cuda::RNS_MODULI[k];
        v[k] = x[k];
        __syncthreads();
        for (int j = 0; j < RNS_MODULI_SIZE - 1; j++) {
            if(k > j){
                int sum = mod_psub(v[k], v[j], m);
                v[k] = mod_mul(sum, cuda::MRC_MULT_INV[j][k], m);
            }
            __syncthreads();
        }
        mr[k] = v[k];
    }

    /*!
     * Computes the most significant digit of the mixed-radix representation of x
     * @param msd - pointer to the result most significant mixed-radix digit (singular)
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE int mrc_pipeline_msd(int * x){
        __shared__ int v[RNS_MODULI_SIZE];
        int k = threadIdx.x;
        int m = cuda::RNS_MODULI[k];
        v[k] = x[k];
        __syncthreads();
        for (int j = 0; j < RNS_MODULI_SIZE - 1; j++) {
            if(k > j){
                int sum = mod_psub(v[k], v[j], m);
                v[k] = mod_mul(sum, cuda::MRC_MULT_INV[j][k], m);
            }
            __syncthreads();
        }
        return (k == 0) * v[RNS_MODULI_SIZE - 1]; // Only thread 0 returns non-zero result
    }

    /*!
     * Compares two RNS numbers using mixed-radix conversion
     * @return 1, if x > y; -1, if x < y; 0, if x = y
     */
    DEVICE_CUDA_FORCEINLINE int mrc_compare_rns(int * x, int * y) {
        int mx[RNS_MODULI_SIZE];
        int my[RNS_MODULI_SIZE];
        cuda::mrc(mx, x);
        cuda::mrc(my, y);
        return cuda::mrs_cmp(mx, my);
    }

    /*!
     * Compares two RNS numbers using the parallel mixed-radix conversion algorithm
     * This routine must be executed by RNS_MODULI_SIZE threads concurrently.
     * Note that all threads returns the same comparison result
     * @return 1, if x > y; -1, if x < y; 0, if x = y
     */
    DEVICE_CUDA_FORCEINLINE int mrc_pipeline_compare_rns(int * x, int * y) {
        __shared__ int mx[RNS_MODULI_SIZE];
        __shared__ int my[RNS_MODULI_SIZE];
        cuda::mrc_pipeline(mx, x);
        cuda::mrc_pipeline(my, y);
        __syncthreads();
        return cuda::mrs_cmp(mx, my);
    }

} //end of namespace

#endif //GRNS_MRC_CUH
