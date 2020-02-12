/*
 *  The set of RNS moduli and other parameters of the GRNS library
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

#define RNS_MODULI_SIZE (128)

#define RNS_PARALLEL_REDUCTION_IDX (64)

#define RNS_MODULI_PRODUCT_LOG2 (3200)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {33590001,33590003,33590005,33590009,33590011,33590017,33590021,33590023,
                          33590027,33590029,33590033,33590041,33590047,33590057,33590059,33590069,
                          33590071,33590077,33590083,33590087,33590093,33590099,33590101,33590107,
                          33590129,33590131,33590153,33590159,33590173,33590177,33590191,33590197,
                          33590201,33590203,33590209,33590213,33590231,33590237,33590239,33590243,
                          33590251,33590257,33590261,33590269,33590273,33590279,33590287,33590299,
                          33590303,33590311,33590329,33590341,33590363,33590369,33590371,33590377,
                          33590387,33590399,33590407,33590429,33590437,33590441,33590449,33590467,
                          33590471,33590483,33590489,33590503,33590507,33590509,33590519,33590527,
                          33590533,33590539,33590549,33590551,33590563,33590567,33590569,33590573,
                          33590581,33590587,33590591,33590593,33590617,33590621,33590629,33590633,
                          33590653,33590659,33590663,33590677,33590693,33590699,33590707,33590717,
                          33590723,33590737,33590743,33590747,33590749,33590761,33590771,33590773,
                          33590779,33590789,33590797,33590801,33590807,33590813,33590819,33590839,
                          33590863,33590867,33590873,33590891,33590897,33590899,33590911,33590927,
                          33590933,33590939,33590941,33590951,33590957,33590959,33590969,33590981};

#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H
