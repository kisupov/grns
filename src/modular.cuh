/*
 *  Modular (modulo m) integer operations, as well as unrolled addition,
 *  subtraction and multiplication in the Residue Number System.
 */

#ifndef GRNS_MODULAR_CUH
#define GRNS_MODULAR_CUH

#include "params.h"
#include "common.cuh"

/**
 * Constants that are computed  or copied to the device memory in rnsbase.cuh
 */
namespace cuda {
    __device__ int RNS_MODULI[RNS_MODULI_SIZE]; // The set of RNS moduli for GPU computing
}


/********************* Integer modulo m operations *********************/


/*!
 * Modulo m addition of x and y using the long data type
 * for intermediate result to avoid overflow.
 * In order to speedup computations, the modulo operation is replaced
 * by multiplication by d = 1 / m.
*/
GCC_FORCEINLINE int mod_add(const int x, const int y, const int m, const double d){
    long r = (long)x + (long)y;
    double quotient = (double) r * d;
    int i = (int) quotient;
    return (int) (r - (long) i * (long) m);
}

/*!
 * Modulo m subtraction of x and y using the long data type
 * for intermediate result to avoid overflow.
 * The subtraction result is not adjusted and may be negative
 */
GCC_FORCEINLINE int mod_sub(const int x, const int y, const int m){
    long r = (long)x - (long)y;
    r = r % (long)m;
    return (int) r;
}

/*!
 * Modulo m subtraction of x and y using the long data type
 * for intermediate result to avoid overflow.
 * Returns the adjusted (non-negative) result.
 */
GCC_FORCEINLINE int mod_psub(const int x, const int y, const int m){
    long r = ((long)x - (long)y + (long)m);
    r = r % (long)m;
    return (int) r;
}

/*!
 * Modulo m multiplication of x and y using the long data type
 * for intermediate result to avoid overflow.
 */
GCC_FORCEINLINE int mod_mul(const int x, const int y, const int m){
    long r = (long)x * (long)y;
    r = r % (long)m;
    return (int)r;
}

/*!
 * Modulo m multiplication of x and y using the long data type
 * for intermediate result to avoid overflow.
 * In order to speedup computations, the modulo operation is replaced
 * by multiplication by d = 1 / m.
*/
GCC_FORCEINLINE int mod_mul(const int x, const int y, const int m, const double d){
    long r = (long)x * (long)y;
    double quotient = (double) r * d;
    int i = (int) quotient;
    return (int) (r - (long) i * (long) m);
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Modulo m addition of x and y using the long data type
     * for intermediate result to avoid overflow.
     */
    DEVICE_CUDA_FORCEINLINE int mod_add(const int x, const int y, const int m){
        long r = (long)x + (long)y;
        r = r % (long)m;
        return (int)r;
    }

    /*!
     * Modulo m subtraction of x and y using the long data type
     * for intermediate result to avoid overflow.
     * The subtraction result is not adjusted and may be negative
     */
    DEVICE_CUDA_FORCEINLINE int mod_sub(const int x, const int y, const int m){
        long r = (long)x - (long)y;
        r = r % (long)m;
        return (int)r;
    }

    /*!
     * Modulo m subtraction of x and y using the long data type
     * for intermediate result to avoid overflow.
     * Returns the adjusted (non-negative) result.
     */
    DEVICE_CUDA_FORCEINLINE int mod_psub(const int x, const int y, const int m){
        long r = ((long)x - (long)y + (long)m);
        r = r % (long)m;
        return (int) r;
    }

    /*!
     * Modulo m multiplication of x and y using the long data type
     * for intermediate result to avoid overflow.
     */
    DEVICE_CUDA_FORCEINLINE int mod_mul(const int x, const int y, const int m){
        long r = (long)x * (long)y;
        r = r % (long)m;
        return (int)r;
    }

} //end of namespace

/********************* Common RNS functions *********************/


/*!
 * Returns true if the RNS number is zero
 */
GCC_FORCEINLINE bool rns_check_zero(int * x) {
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
      if(x[i] != 0){
          return false;
      }
    }
    return true;
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Returns true if the RNS number is zero
     */
    DEVICE_CUDA_FORCEINLINE bool rns_check_zero(int * x) {
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            if(x[i] != 0){
                return false;
            }
        }
        return true;
    }

} //end of namespace

/********************* Modular arithmetic over RNS numbers *********************/

/*!
 *  Multiplication of two RNS numbers.
 */
GCC_FORCEINLINE void rns_mul(int * result, int * x, int * y){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        result[i] = mod_mul(x[i], y[i], RNS_MODULI[i], 1.0/RNS_MODULI[i]);
    }
}

/*!
 * Addition of two RNS numbers.
 */
GCC_FORCEINLINE void rns_add(int * result, int * x, int * y){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        result[i] = mod_add(x[i], y[i], RNS_MODULI[i], 1.0/RNS_MODULI[i]);
    }
}

/*!
 * Unrolled subtraction of two RNS numbers.
 */
GCC_FORCEINLINE void rns_sub(int * result, int * x, int * y){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        result[i] = mod_psub(x[i], y[i], RNS_MODULI[i]);
    }
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
    * Assign the value of an RNS number to another RNS number
    */
    DEVICE_CUDA_FORCEINLINE void rns_set(int * result, int * src){
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = src[i];
        }
    }

    /*!
     * Multiplication of two RNS numbers.
     */
    DEVICE_CUDA_FORCEINLINE void rns_mul(int * result, int * x, int * y){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = cuda::mod_mul(x[i], y[i], moduli[i]);
        }
    }

    /*!
      * Addition of two RNS numbers.
      */
    DEVICE_CUDA_FORCEINLINE void rns_add(int * result, int * x, int * y){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = cuda::mod_add(x[i], y[i], moduli[i]);
        }
    }

    /*!
     * Subtraction of two RNS numbers.
     */
    DEVICE_CUDA_FORCEINLINE void rns_sub(int * result, int * x, int * y){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = cuda::mod_psub(x[i], y[i], moduli[i]);
        }
    }

} //end of namespace


#endif //GRNS_MODULAR_CUH
