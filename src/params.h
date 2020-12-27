/*
 *  The set of RNS moduli and other parameters of GRNS
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

/*
 * Size of the RNS moduli set
 */
#define RNS_MODULI_SIZE (8)

/*
 * Initial index for parallel reduction in loops over the RNS moduli.
 * The largest power of two which strictly less than RNS_MODULI_SIZE
 */
#define RNS_PARALLEL_REDUCTION_IDX (4)

/*
 * Binary logarithm of the full RNS moduli product
 */
#define RNS_MODULI_PRODUCT_LOG2 (242)

/*
 * Maximal power-of-two for one scaling step in the RNS system.
 * It should be such that operations modulo 2^RNS_P2_SCALING_THRESHOLD are performed efficiently.
 */
#define RNS_P2_SCALING_THRESHOLD (30)

/*
 * Upper bound for the relative forward error of an RNS interval evaluation
 */
#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

/*
 * The set of RNS moduli
 */
#define RNS_MODULI_VALUES {1283742825,1283742827,1283742829,1283742833,1283742839,1283742841,1283742847,1283742851}

constexpr int RNS_MODULI[RNS_MODULI_SIZE] = RNS_MODULI_VALUES;

/*
 * Specifies whether to use std::fma to compute (x * y) + z
 */
#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H
