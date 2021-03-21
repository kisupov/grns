/*
 *  Comparing the magnitude of RNS numbers using floating-point intervals
 */

#ifndef GRNS_CMP_CUH
#define GRNS_CMP_CUH

#include "rnseval.cuh"
#include "mrc.cuh"

/*!
 * Given two integers x and y such that 0 \le x,y < M, represented as x = (x1 ,x2,...,xn) and y = (y1 ,y2,...,yn),
 * this routine returns:
 *  0, if x = y
 *  1, if x > y
 * -1, if x < y
 * @param x - pointer to the number in the RNS
 * @param y - pointer to the number in the RNS
 */
GCC_FORCEINLINE int rns_cmp(int *x, int *y) {
    interval_t ex; //Interval evaluation of x
    interval_t ey; //Interval evaluation of y
    rns_eval_compute(&ex.low, &ex.upp, x);
    rns_eval_compute(&ey.low, &ey.upp, y);
    if(er_ucmp(&ex.low, &ey.upp) > 0){
        return 1;
    }
    if(er_ucmp(&ey.low, &ex.upp) > 0){
        return -1;
    }
    bool equals = true;
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        if(x[i] != y[i]){
            equals = false;
            break;
        }
    }
    return equals ? 0 : mrc_compare_rns(x, y);
}


/*
 * GPU functions
 */
namespace cuda {
    /*!
     * Given two integers x and y such that 0 \le x,y < M, represented as x = (x1 ,x2,...,xn) and y = (y1 ,y2,...,yn),
     * this routine returns:
     *  0, if x = y
     *  1, if x > y
     * -1, if x < y
     * @param x - pointer to the number in the RNS
     * @param y - pointer to the number in the RNS
     */
    DEVICE_CUDA_FORCEINLINE int rns_cmp(int *x, int *y) {
        interval_t ex; //Interval evaluation of x
        interval_t ey; //Interval evaluation of y
        cuda::rns_eval_compute(&ex.low, &ex.upp, x);
        cuda::rns_eval_compute(&ey.low, &ey.upp, y);
        if(cuda::er_ucmp(&ex.low, &ey.upp) > 0){
            return 1;
        }
        if(cuda::er_ucmp(&ey.low, &ex.upp) > 0){
            return -1;
        }
        bool equals = true;
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            if(x[i] != y[i]){
                equals = false;
                break;
            }
        }
        return equals ? 0 : cuda::mrc_compare_rns(x, y);
    }

    /*!
     * Given two integers x and y such that 0 \le x,y < M, represented as x = (x1 ,x2,...,xn) and y = (y1 ,y2,...,yn),
     * this routine returns:
     *  0, if x = y
     *  1, if x > y
     * -1, if x < y
     * This routine must be executed by RNS_MODULI_SIZE threads concurrently.
     * Note that all threads returns the same comparison result
     * @param x - pointer to the number in the RNS
     * @param y - pointer to the number in the RNS
     */
    DEVICE_CUDA_FORCEINLINE int rns_cmp_parallel(int *x, int *y) {
        __shared__ interval_t xeval; //Interval evaluation of x
        __shared__ interval_t yeval; //Interval evaluation of y

        cuda::rns_eval_compute_parallel(&xeval.low, &xeval.upp, x);
        cuda::rns_eval_compute_parallel(&yeval.low, &yeval.upp, y);
        __syncthreads();
        if(cuda::er_ucmp(&xeval.low, &yeval.upp) > 0){
            return 1;
        }
        if(cuda::er_ucmp(&yeval.low, &xeval.upp) > 0){
            return -1;
        }
        bool equals = true;
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            if(x[i] != y[i]){
                equals = false;
                break;
            }
        }
        __syncthreads();
        return equals ? 0 : cuda::mrc_pipeline_compare_rns(x, y);
    }

}

#endif  //GRNS_CMP_CUH