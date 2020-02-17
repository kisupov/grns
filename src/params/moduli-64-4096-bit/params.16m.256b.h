/*
 *  The set of RNS moduli and other parameters of the GRNS library
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

#define RNS_MODULI_SIZE (16)

#define RNS_PARALLEL_REDUCTION_IDX (8)

#define RNS_MODULI_PRODUCT_LOG2 (256)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {65599,65601,65603,65605,65609,65611,65617,65621,
                          65623,65627,65629,65633,65641,65647,65651,65657};

#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H