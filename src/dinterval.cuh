/*
 *  Simulation of double precision interval arithmetic operations
 *  with fixed rounding mode (rounding to nearest) according to
 *  Algorithm 4 (BoundNear1) proposed in:
 *  Siegfried M. Rump, Takeshi Ogita, Yusuke Morikura, Shin'ichi Oishi,
 *  Interval arithmetic with fixed rounding mode //
 *  Nonlinear Theory and Its Applications, IEICE, 2016, Volume 7, Issue 3, Pages 362-373,
 *  https://doi.org/10.1587/nolta.7.362
 *
 *  Note that the operations provided are not IEEE 754 compatible.
 *  Also note that these operations are cross-platform, but are generally slower than the dynamic switching
 *  of rounding modes using a floating point control word.
 */

#ifndef GRNS_DINTERVAL_CUH
#define GRNS_DINTERVAL_CUH

#include "bitwise.cuh"
#include "params.h"
#include <cmath>

/*
 * Computes (x * y) + z
 */
GCC_FORCEINLINE double fmac(const double x, const double y, const double z){
    #if EMPLOY_STD_FMA
        return std::fma(x, y, z);
    #else
        return x*y+z;
    #endif
}

/*!
 *	Computing the predecessor of a + b (addition with rounding downwards)
 *	Returns pred(a + b) that less than or equal to (a + b)
 */
GCC_FORCEINLINE double dadd_rd(const double a, const double b){
    double c = a + b;
    return c - fmac(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Computing the successor of a + b (addition with rounding upwards).
 *	Returns succ(a + b) greater than or equal to (a + b)
 */
GCC_FORCEINLINE double dadd_ru(const double a, const double b){
    double c = a + b;
    return c + fmac(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Enclosing interval of a + b. Returns the interval [low, upp] that includes (a + b),
 *	where low and upp are the pointers to the result lower and upper bounds, respectively.
 */
GCC_FORCEINLINE void dadd_rdu(double * low, double * upp, const double a, const double b){
    double c = a + b;
    double e = fmac(DBL_PHI1, std::abs(c), DBL_ETA);
    *low = c - e;
    *upp = c + e;
}

/*!
 *	Computing the predecessor of a + b (subtraction with rounding downwards)
 *	Returns pred(a - b) that less than or equal to (a - b)
 */
GCC_FORCEINLINE double dsub_rd(const double a, const double b){
    double c = a - b;
    return c - fmac(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Computing the successor of a - b (subtraction with rounding upwards)
 *	Returns succ(a - b) greater than or equal to (a - b)
 */
GCC_FORCEINLINE double dsub_ru(const double a, const double b){
    double c = a - b;
    return c + fmac(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Enclosing interval of a - b. Returns the interval [low, upp] that includes (a - b),
 *	where low and upp are the pointers to the result lower and upper bounds, respectively.
 */
GCC_FORCEINLINE void dsub_rdu(double * low, double * upp, const double a, const double b){
    double c = a - b;
    double e = fmac(DBL_PHI1, std::abs(c), DBL_ETA);
    *low = c - e;
    *upp = c + e;
}

/*!
 *	Computing the predecessor of a * b (multiplication with rounding downwards)
 *	Returns pred(a * b) that less than or equal to (a * b)
 */
GCC_FORCEINLINE double dmul_rd(const double a, const double b){
    double c = a * b;
    return c - fmac(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Computing the successor of a * b (multiplication with rounding upwards)
 *	Returns succ(a * b) greater than or equal to (a * b)
 */
GCC_FORCEINLINE double dmul_ru(const double a, const double b){
    double c = a * b;
    return c + fmac(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Enclosing interval of a * b. Returns the interval [low, upp] that includes (a * b),
 *	where low and upp are the pointers to the result lower and upper bounds, respectively.
 */
GCC_FORCEINLINE void dmul_rdu(double * low, double * upp, const double a, const double b){
    double c = a * b;
    double e = fmac(DBL_PHI1, std::abs(c), DBL_ETA);
    *low = c - e;
    *upp = c + e;
}

/*!
 *	Computing the predecessor of a / b (division with rounding downwards)
 *	Returns pred(a / b) that less than or equal to (a / b)
 */
GCC_FORCEINLINE double ddiv_rd(const double a, const double b){
    double c = a / b;
    return c - fmac(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Computing the successor of a / b (division with rounding upwards)
 *	Returns succ(a / b) greater than or equal to (a / b)
 */
GCC_FORCEINLINE double ddiv_ru(const double a, const double b){
    double c = a / b;
    return c + fmac(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Enclosing interval of a / b. Returns the interval [low, upp] that includes (a/b),
 *	where low and upp are the pointers to the result lower and upper bounds, respectively.
 */
GCC_FORCEINLINE void ddiv_rdu(double * low, double * upp, const double a, const double b){
    double c = a / b;
    double e = fmac(DBL_PHI1, std::abs(c), DBL_ETA);
    *low = c - e;
    *upp = c + e;
}

#endif //GRNS_DINTERVAL_CUH
