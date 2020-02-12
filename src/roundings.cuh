/*
 *  Functions for changing IEEE 754 rounding modes with different compilers
 */

#ifndef GRNS_ROUNDINGS_CUH
#define GRNS_ROUNDINGS_CUH

#if defined(__ICC) || defined(__INTEL_COMPILER) || defined(__GNUG__)
#include <xmmintrin.h>
//Masks for change rounding modes (for Intel compiler only)
unsigned int _mxcsr_up = _MM_MASK_MASK | _MM_ROUND_UP;
unsigned int _mxcsr_down = _MM_MASK_MASK | _MM_ROUND_DOWN;
unsigned int _mxcsr_n = _MM_MASK_MASK;
unsigned int _mxcsr_zero = _MM_MASK_MASK | _MM_ROUND_TOWARD_ZERO;
#elif defined(_MSC_VER)
#pragma STDC FENV_ACCESS ON
#endif

// https://msdn.microsoft.com/en-us/library/y70z2105(v=vs.71).aspx
// http://www.club155.ru/x86internalreg-fpucw

void round_nearest_mode() {
#if defined(__ICC) || defined(__INTEL_COMPILER)
    __asm { ldmxcsr _mxcsr_n }
#elif defined(_MSC_VER)
    short x;
    __asm
    {
        mov x, 1101111111b
        fldcw x
    }
#elif defined(__GNUG__)
    asm (
    "ldmxcsr %0" : : "m" (_mxcsr_n)
    );
#endif
}

void round_up_mode() {
#if defined(__ICC) || defined(__INTEL_COMPILER)
    __asm { ldmxcsr _mxcsr_up }
#elif defined(_MSC_VER)
    short x;
    __asm
    {
        mov x, 101101111111b
        fldcw x
    }
#elif defined(__GNUG__)
    asm (
    "ldmxcsr %0" : : "m" (_mxcsr_up)
    );
#endif
}

void round_down_mode() {
#if defined(__ICC) || defined(__INTEL_COMPILER)
    __asm { ldmxcsr _mxcsr_down }
#elif defined(_MSC_VER)
    short x;
    __asm
    {
        mov x, 011101111111b
        fldcw x
    }
#elif defined(__GNUG__)
    {
        asm (
        "ldmxcsr %0" : : "m" (_mxcsr_down)
        );
    }
#endif
}

void round_zero_mode() {
#if defined(__ICC) || defined(__INTEL_COMPILER)
    __asm { ldmxcsr _mxcsr_zero }
#elif defined(_MSC_VER)
    short x;
    __asm
    {
        mov x, 111101111111b
        fldcw x
    }
#elif defined(__GNUG__)
    {
        asm (
        "ldmxcsr %0" : : "m" (_mxcsr_zero)
        );
    }
#endif
}

#endif //GRNS_ROUNDINGS_CUH
