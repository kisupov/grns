/*
 * Data structures for representing extended-range floating-point intervals
 */

#ifndef GRNS_TYPES_CUH
#define GRNS_TYPES_CUH

#include "params.h"

/*!
 * Extended-range floating-point representation
 */
typedef struct {
    double frac; // Significand
    long exp;    // Exponent
} er_float_t;

typedef er_float_t * er_float_ptr;


/*!
 * Interval evaluation for the fractional representation of a number represented in the Residue Number System (RNS).
 * We called this 'RNS interval evaluation'
 */
typedef struct {
    er_float_t low; // Lower bound
    er_float_t upp; // Upper bound
} interval_t;

typedef interval_t * interval_ptr;

#endif //GRNS_TYPES_CUH