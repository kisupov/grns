/*
 *  Implementation of the division operation in RNS using floating-point intervals
 */

#ifndef GRNS_DIV_CUH
#define GRNS_DIV_CUH

#include "rnseval.cuh"

/*!
 * Given two integers x, 0 \le x < M and d, 1 \le d < M, represented as x = (x1 ,x2,...,xn) and d = (d1 ,d2,...,dn ),
 * this routine performs the division operation, i.e., computes the quotient q and the remainder r such that x = qd + r and 0 \le r < d.
 * @param q - pointer to the quotient in the RNS
 * @param r - pointer to the remainder in the RNS
 * @param x - pointer to the  dividend in the RNS
 * @param d - pointer to the divisor in the RNS
 */
GCC_FORCEINLINE void rns_div(int *q, int *r, int *x, int *d) {
    int pq[RNS_MODULI_SIZE]; //Partial quotient
    er_float_t flq; //The result of floating-point division
    interval_t deval; //Interval evaluation of d
    interval_t reval; //Interval evaluation of r

    //Set the initial values of q and r
    memset(q, 0, RNS_MODULI_SIZE * sizeof(int));
    memcpy(r, x, RNS_MODULI_SIZE * sizeof(int));

    //Compute the RNS interval evaluations for d and r
    rns_eval_compute(&deval.low, &deval.upp, d);
    rns_eval_compute(&reval.low, &reval.upp, r);

    //Main division loop
    while (er_ucmp(&reval.low, &deval.upp) >= 0) {
        er_div_rd(&flq, &reval.low, &deval.upp); //Floating-point division
        memcpy(pq, RNS_POW2[flq.exp], RNS_MODULI_SIZE * sizeof(int)); //Fetch the partial quotient, pq, from the LUT
        rns_add(q, q, pq); //q = q + pq
        rns_mul(pq, pq, d); //pq = d * pq
        rns_sub(r, r, pq); // r = r - pq
        rns_eval_compute(&reval.low, &reval.upp, r);
    }
    //Final adjustment
    if (er_ucmp(&reval.upp, &deval.low) >= 0) { //Ambiguity, use MRC
        int dmr[RNS_MODULI_SIZE]; //for mixed-radix representation of d
        bool rged = true; // r \ge d
        perform_mrc(dmr, d);
        perform_mrc(pq, r);
        for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
            if (pq[i] < dmr[i]) {
                rged = false; //r < d, no adjustment required
                break;
            } else if (pq[i] > dmr[i]) {
                break;
            }
        }
        if (rged) {
            rns_add(q, q, RNS_ONE);
            rns_sub(r, r, d);
        }
    }
}

/*!
 * This is an improved version of the rns_div algorithm.
 * It eliminates multiple recalculations of the interval evaluation
 */
GCC_FORCEINLINE void rns_div_fast(int *q, int *r, int *x, int *d) {
    int pq[RNS_MODULI_SIZE]; //Partial quotient
    er_float_t flq; //The result of floating-point division
    interval_t deval; //Interval evaluation of d
    interval_t reval; //Interval evaluation of r

    //Set the initial values of q and r
    memset(q, 0, RNS_MODULI_SIZE * sizeof(int));
    memcpy(r, x, RNS_MODULI_SIZE * sizeof(int));

    //Compute the RNS interval evaluations for d and r
    rns_eval_compute(&deval.low, &deval.upp, d);
    rns_eval_compute(&reval.low, &reval.upp, r);

    //Main division loop
    while (er_ucmp(&reval.low, &deval.upp) >= 0) {
        er_div_rd(&flq, &reval.low, &deval.upp); //Floating-point division
        memset(pq, 0, RNS_MODULI_SIZE * sizeof(int)); //Reset the total partial quotient
        while (flq.exp >= 0 && flq.frac > 0){
            rns_add(pq,pq,RNS_POW2[flq.exp]); //Fetch the partial quotient from the LUT and add it to pq
            flq.frac = dsub_rd(flq.frac, 1.0); //flq = flq - 2^e
            er_adjust(&flq); //Normalization flq to place flq.frac in the interval 1 <= flq.frac < 2
        }
        rns_add(q, q, pq); //q = q + pq
        rns_mul(pq, pq, d); //pq = d * pq
        rns_sub(r, r, pq); // r = r - pq
        rns_eval_compute(&reval.low, &reval.upp, r);
    }
    //Final adjustment
    if (er_ucmp(&reval.upp, &deval.low) >= 0) { //Ambiguity, use MRC
        int dmr[RNS_MODULI_SIZE]; //for mixed-radix representation of d
        bool rged = true; // r \ge d
        perform_mrc(dmr, d);
        perform_mrc(pq, r);
        for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
            if (pq[i] < dmr[i]) {
                rged = false; //r < d, no adjustment required
                break;
            } else if (pq[i] > dmr[i]) {
                break;
            }
        }
        if (rged) {
            rns_add(q, q, RNS_ONE);
            rns_sub(r, r, d);
        }
    }
}

/*
 * GPU functions
 */
namespace cuda {
    /*!
     * Given two integers x, 0 \le x < M and d, 1 \le d < M, represented as x = (x1 ,x2,...,xn) and d = (d1 ,d2,...,dn ),
     * this routine performs the division operation, i.e., computes the quotient q and the remainder r such that x = qd + r and 0 \le r < d.
     * @param q - pointer to the quotient in the RNS
     * @param r - pointer to the remainder in the RNS
     * @param x - pointer to the  dividend in the RNS
     * @param d - pointer to the divisor in the RNS
     */
    DEVICE_CUDA_FORCEINLINE void rns_div(int *q, int *r, int *x, int *d) {
        int pq[RNS_MODULI_SIZE]; //Partial quotient
        er_float_t flq; //The result of floating-point division
        interval_t deval; //Interval evaluation of d
        interval_t reval; //Interval evaluation of r

        //Set the initial values of q and r
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            q[i] = 0;
            r[i] = x[i];
        }

        //Compute the RNS interval evaluations for d and r
        cuda::rns_eval_compute(&deval.low, &deval.upp, d);
        cuda::rns_eval_compute(&reval.low, &reval.upp, r);

        //Main division loop
        while (cuda::er_ucmp(&reval.low, &deval.upp) >= 0) {
            cuda::er_div_rd(&flq, &reval.low, &deval.upp); //Floating-point division
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                pq[i] = cuda::RNS_POW2[flq.exp][i]; //Fetch the partial quotient, pq, from the LUT
                q[i] = cuda::mod_add(q[i], pq[i], cuda::RNS_MODULI[i]); //q = q + pq
                pq[i] = cuda::mod_mul(pq[i], d[i], cuda::RNS_MODULI[i]); //pq = d * pq
                r[i] = cuda::mod_psub(r[i], pq[i], cuda::RNS_MODULI[i]); //r = r - pq
            }
            cuda::rns_eval_compute(&reval.low, &reval.upp, r);
        }
        //Final adjustment
        if (cuda::er_ucmp(&reval.upp, &deval.low) >= 0) { //Ambiguity, use MRC
            int dmr[RNS_MODULI_SIZE]; //for mixed-radix representation of d
            bool rged = true; // r \ge d
            cuda::perform_mrc(dmr, d);
            cuda::perform_mrc(pq, r);
            for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
                if (pq[i] < dmr[i]) {
                    rged = false; //r < d, no adjustment required
                    break;
                } else if (pq[i] > dmr[i]) {
                    break;
                }
            }
            if (rged) {
                for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                    q[i] = cuda::mod_add(q[i], 1, cuda::RNS_MODULI[i]);
                    r[i] = cuda::mod_psub(r[i], d[i], cuda::RNS_MODULI[i]);
                }
            }
        }
    }

    /*!
     * Given two integers x, 0 \le x < M and d, 1 \le d < M, represented as x = (x1 ,x2,...,xn) and d = (d1 ,d2,...,dn ),
     * this routine in a parallel manner performs the division operation, i.e., computes the quotient q and the remainder r such that x = qd + r and 0 \le r < d.
     * This routine must be executed by RNS_MODULI_SIZE threads concurrently.
     * @param q - pointer to the quotient in the RNS
     * @param r - pointer to the remainder in the RNS
     * @param x - pointer to the  dividend in the RNS
     * @param d - pointer to the divisor in the RNS
     */
    DEVICE_CUDA_FORCEINLINE void rns_div_parallel(int *q, int *r, int *x, int *d) {
        int pq; //Partial quotient
        int modulus = cuda::RNS_MODULI[threadIdx.x];
        int ld = d[threadIdx.x]; //Digit of divisor
        er_float_t flq; //The result of floating-point division
        __shared__ interval_t deval; //Interval evaluation of d
        __shared__ interval_t reval; //Interval evaluation of r
        __shared__ int dmr[RNS_MODULI_SIZE]; //for mixed-radix representation of d
        __shared__ int rmr[RNS_MODULI_SIZE]; //for mixed-radix representation of r
        __shared__ bool rged; // r \ge d

        //Set the initial values of q and r
        q[threadIdx.x] = 0;
        r[threadIdx.x] = x[threadIdx.x];

        //Compute the RNS interval evaluations for d and r
        cuda::rns_eval_compute_parallel(&deval.low, &deval.upp, d);
        cuda::rns_eval_compute_parallel(&reval.low, &reval.upp, r);
        __syncthreads();

        //Main division loop
        while (cuda::er_ucmp(&reval.low, &deval.upp) >= 0) {
            cuda::er_div_rd(&flq, &reval.low, &deval.upp); //Floating-point division
            pq = cuda::RNS_POW2[flq.exp][threadIdx.x]; //Fetch the partial quotient, pq, from the LUT
            q[threadIdx.x] = cuda::mod_add(q[threadIdx.x], pq, modulus); //q = q + pq
            pq = cuda::mod_mul(pq, ld, modulus); //pq = d * pq
            r[threadIdx.x] = cuda::mod_psub(r[threadIdx.x], pq, modulus); //r = r - pq
            cuda::rns_eval_compute_parallel(&reval.low, &reval.upp, r);
            __syncthreads();
        }
        //Final adjustment
        if (cuda::er_ucmp(&reval.upp, &deval.low) >= 0) { //Ambiguity, use MRC
            cuda::perform_mrc_parallel(dmr, d);
            cuda::perform_mrc_parallel(rmr, r);
            if (threadIdx.x == 0) {
                rged = true;
                for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
                    if (rmr[i] < dmr[i]) {
                        rged = false; //r < d, no adjustment required
                        break;
                    } else if (rmr[i] > dmr[i]) {
                        break;
                    }
                }
            }
            __syncthreads();
            if (rged) {
                q[threadIdx.x] = cuda::mod_add(q[threadIdx.x], 1, modulus);
                r[threadIdx.x] = cuda::mod_psub(r[threadIdx.x], ld, modulus);
            }
        }
    }

    /*!
     * This is an improved version of the cuda::rns_div_parallel algorithm.
     * It eliminates multiple recalculations of the interval evaluation
     */
    DEVICE_CUDA_FORCEINLINE void rns_div_parallel_fast(int *q, int *r, int *x, int *d) {
        int pq; //Partial quotient
        int modulus = cuda::RNS_MODULI[threadIdx.x];
        int ld = d[threadIdx.x]; //Digit of divisor
        er_float_t flq; //The result of floating-point division
        __shared__ interval_t deval; //Interval evaluation of d
        __shared__ interval_t reval; //Interval evaluation of r
        __shared__ int dmr[RNS_MODULI_SIZE]; //for mixed-radix representation of d
        __shared__ int rmr[RNS_MODULI_SIZE]; //for mixed-radix representation of d
        __shared__ bool rged; // r \ge d

        //Set the initial values of q and r
        q[threadIdx.x] = 0;
        r[threadIdx.x] = x[threadIdx.x];

        //Compute the RNS interval evaluations for d and r
        cuda::rns_eval_compute_parallel(&deval.low, &deval.upp, d);
        cuda::rns_eval_compute_parallel(&reval.low, &reval.upp, r);
        __syncthreads();

        //Main division loop
        while (cuda::er_ucmp(&reval.low, &deval.upp) >= 0) {
            cuda::er_div_rd(&flq, &reval.low, &deval.upp); //Floating-point division
            pq = 0.0; //Reset the total partial quotient
            while(flq.frac > 0 && flq.exp >=0){
                pq = cuda::mod_add(pq, cuda::RNS_POW2[flq.exp][threadIdx.x], modulus); //Fetch the partial quotient from the LUT and add it to pq
                flq.frac = __dsub_rd(flq.frac, 1.0); //flq = flq - 2^e
                cuda::er_adjust(&flq); //Normalization flq to place flq.frac in the interval 1 <= flq.frac < 2
            }
            q[threadIdx.x] = cuda::mod_add(q[threadIdx.x], pq, modulus); //q = q + pq
            pq = cuda::mod_mul(pq, ld, modulus); //pq = d * pq
            r[threadIdx.x] = cuda::mod_psub(r[threadIdx.x], pq, modulus); //r = r - pq
            cuda::rns_eval_compute_parallel(&reval.low, &reval.upp, r);
            __syncthreads();
        }
        //Final adjustment
        if (cuda::er_ucmp(&reval.upp, &deval.low) >= 0) { //Ambiguity, use MRC
            cuda::perform_mrc_parallel(dmr, d);
            cuda::perform_mrc_parallel(rmr, r);
            if (threadIdx.x == 0) {
                rged = true;
                for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
                    if (rmr[i] < dmr[i]) {
                        rged = false; //r < d, no adjustment required
                        break;
                    } else if (rmr[i] > dmr[i]) {
                        break;
                    }
                }
            }
            __syncthreads();
            if (rged) {
                q[threadIdx.x] = cuda::mod_add(q[threadIdx.x], 1, modulus);
                r[threadIdx.x] = cuda::mod_psub(r[threadIdx.x], ld, modulus);
            }
        }
    }

}

#endif  //GRNS_DIV_CUH