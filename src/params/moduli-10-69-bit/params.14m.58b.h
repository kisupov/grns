/*
 *  The set of RNS moduli and other parameters of the GRNS library
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

#define RNS_MODULI_SIZE (14)

#define RNS_PARALLEL_REDUCTION_IDX (8)

#define RNS_MODULI_PRODUCT_LOG2 (58)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {3,5,7,11,13,17,19,23,29,31,37,41,43,47};

#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H