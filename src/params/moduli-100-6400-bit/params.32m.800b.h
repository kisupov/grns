/*
 *  The set of RNS moduli and other parameters of the GRNS library
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

#define RNS_MODULI_SIZE (32)

#define RNS_PARALLEL_REDUCTION_IDX (16)

#define RNS_MODULI_PRODUCT_LOG2 (800)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {33742825,33742827,33742829,33742831,33742837,33742841,33742843,33742847,
                          33742849,33742853,33742859,33742861,33742867,33742871,33742873,33742883,
                          33742889,33742897,33742901,33742903,33742913,33742921,33742931,33742937,
                          33742939,33742949,33742957,33742963,33742967,33742969,33742981,33742987};

#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H
