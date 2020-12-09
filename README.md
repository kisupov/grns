# GRNS: A CUDA Library for High-Performance Computing in Residue Number System (RNS)
###### Version 1.1.4, released 2020-12-18

GRNS supports the following calculations in the [residue number system](https://en.wikipedia.org/wiki/Residue_number_system) on CUDA-compatible graphics processors:

1. Non-modular (aka inter-modulo) operations that are difficult for RNS:

    * magnitude comparison (`/src/rnscmp.cuh`)
    * general integer division (`/src/rnsdiv.cuh`)
    * power-of-two scaling (`/src/rnsscal.cuh`)

    Efficient execution of these operations relies on finite precision floating-point interval evaluations (`/src/rnseval.cuh`). See [this paper](https://dx.doi.org/10.1109/ACCESS.2020.2982365) for further details. Both serial (single-threaded) and parallel (n-threaded) GPU implementations of the above operations are available. 

2. Arithmetic operations over large integers represented in the RNS (`/src/mpint.cuh`). A multiple-precision integer is represented by the sign, the significand in RNS, 
and the interval evaluation of the significand. Four basic arithmetic operations 
(addition, subtraction, multiplication, and division) are supported for CPU and CUDA. 
For samples of usage, see `/tests/test_verify_mpint.cu` and `/tests/test_perf_mpint.cu`.

3. Data-parallel primitives operating in the RNS domain:

    * finding the maximum element of an array of RNS numbers (`/src/dpp/rnsmax.cuh`)

GRNS is designed for large RNS dynamic ranges, which significantly exceed the usual precision of computers 
(say hundreds and thousands of bits). There are no special restrictions on the moduli sets and 
the magnitude of numbers in RNS representation.

The library can be freely used in various software-based RNS applications, e.g. RSA, Diffie-Hellman, and Elliptic curves. 
Currently, the algorithms implemented in GRNS are used in a multiple-precision GPU accelerated BLAS library; 
see https://github.com/kisupov/mpres-blas.

At the initialization phase, GRNS relies on two open source libraries.
These are the GNU Multiple Precision Library (GMP) and the GNU MPFR Library (MPFR).
For the division benchmark, the CAMPARY library is additionally required.

### Details and notes

1. GRNS is intended for Linux and the GCC compiler. Some manipulations have to be done to run it in Windows.

2. The set of RNS moduli used in defined in `src/params.h`. You can define any arbitrary set of moduli. Several pre-generated sets of moduli that provide various dynamic ranges, from relatively small to cryptographic sizes, are located in the `src/params/` folder. Just replace the content of `src/params.h` with the content of the file you want to use.

3. When using large moduli (like `1283742825`), make sure your system uses LP64 programming model ('long, pointers are 64-bit').  Fortunately, all modern 64-bit Unix systems use LP64.

### References

1. K. Isupov, "Using floating-point intervals for non-modular computations in residue number system," IEEE Access, vol. 8, pp. 58603-58619, 2020, doi 10.1109/ACCESS.2020.2982365.


*Link: http://github.com/kisupov/grns*

*Copyright 2019, 2020 by Konstantin Isupov.*
