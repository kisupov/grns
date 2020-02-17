/*
 *  The set of RNS moduli and other parameters of the GRNS library
 */

#ifndef GRNS_PARAMS_H
#define GRNS_PARAMS_H

#define RNS_MODULI_SIZE (256)

#define RNS_PARALLEL_REDUCTION_IDX (128)

#define RNS_MODULI_PRODUCT_LOG2 (6400)

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
                          33590933,33590939,33590941,33590951,33590957,33590959,33590969,33590981,
                          33590983,33590987,33591043,33591049,33591059,33591067,33591083,33591091,
                          33591097,33591101,33591113,33591121,33591137,33591139,33591143,33591149,
                          33591157,33591161,33591163,33591191,33591199,33591209,33591211,33591223,
                          33591253,33591263,33591277,33591281,33591289,33591319,33591329,33591331,
                          33591343,33591359,33591379,33591401,33591403,33591407,33591409,33591413,
                          33591427,33591433,33591443,33591451,33591463,33591469,33591479,33591491,
                          33591497,33591499,33591517,33591521,33591527,33591533,33591539,33591553,
                          33591577,33591581,33591589,33591617,33591619,33591629,33591637,33591641,
                          33591647,33591653,33591659,33591667,33591673,33591683,33591697,33591703,
                          33591721,33591731,33591737,33591739,33591743,33591749,33591757,33591763,
                          33591769,33591787,33591797,33591809,33591821,33591839,33591841,33591851,
                          33591853,33591863,33591869,33591871,33591911,33591917,33591919,33591923,
                          33591931,33591953,33591959,33591977,33591979,33592001,33592007,33592021,
                          33592033,33592037,33592049,33592061,33592081,33592087,33592093,33592099,
                          33592103,33592109,33592123,33592129,33592147,33592177,33592183,33592193,
                          33592211,33592219,33592231,33592241,33592253,33592261,33592267,33592271};

#define EMPLOY_STD_FMA false

#endif  //GRNS_PARAMS_H