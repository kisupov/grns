/*
 *  The set of RNS moduli and other parameters of the GRNS library
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

#define RNS_MODULI_SIZE (4)

#define RNS_PARALLEL_REDUCTION_IDX (2)

#define RNS_MODULI_PRODUCT_LOG2 (64)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {65947,65949,65951,65953};

#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H
