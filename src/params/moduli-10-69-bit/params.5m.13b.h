/*
 *  The set of RNS moduli and other parameters of the GRNS library
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

#define RNS_MODULI_SIZE (5)

#define RNS_PARALLEL_REDUCTION_IDX (4)

#define RNS_MODULI_PRODUCT_LOG2 (13)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {3,5,7,11,13};

#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H