/*
 *  The set of RNS moduli and other parameters of the GRNS library
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

#define RNS_MODULI_SIZE (64)

#define RNS_PARALLEL_REDUCTION_IDX (32)

#define RNS_MODULI_PRODUCT_LOG2 (1600)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {33742825,33742827,33742829,33742831,33742837,33742841,33742843,33742847,
                          33742849,33742853,33742859,33742861,33742867,33742871,33742873,33742883,
                          33742889,33742897,33742901,33742903,33742913,33742921,33742931,33742937,
                          33742939,33742949,33742957,33742963,33742967,33742969,33742981,33742987,
                          33742991,33742997,33742999,33743009,33743023,33743027,33743029,33743033,
                          33743041,33743051,33743053,33743057,33743063,33743071,33743077,33743081,
                          33743093,33743117,33743119,33743131,33743137,33743141,33743161,33743167,
                          33743173,33743179,33743189,33743197,33743201,33743207,33743209,33743219};

#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H
