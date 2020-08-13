# Library for computations in Residue Number System (RNS) using CUDA-enabled GPUs (GRNS)
###### Version 1.1.2, released 2020-08-13

The following time-consuming non-modular (aka inter-modulo) operations in [RNS](https://en.wikipedia.org/wiki/Residue_number_system) are implemented in addition to simple addition, subtraction, and multiplication:

* magnitude comparison (`/src/rnscmp.cuh`)
* general integer division (`/src/rnsdiv.cuh`)
* power-of-two scaling (`/src/rnsscal.cuh`)

To perform these operations, the interval evaluation of the fractional representation of an RNS number is computed (see `/src/rnseval.cuh`).
Both serial (single-threaded) and parallel (n-threaded) implementations of the above operations are available.

GRNS is designed for large RNS dynamic ranges, which significantly exceed the usual precision of computers 
(say hundreds and thousands of bits). There are no special restrictions on the moduli sets and 
the magnitude of numbers in RNS representation. Some predefined moduli sets are located in `/src/params`.

Since version 1.1.0, GRNS supports multiple-precision integer arithmetic based on the residue number system 
(see `mpint.cuh`). A multiple-precision integer is represented by the sign, the significand in RNS, 
and the interval evaluation of the significand. Four basic arithmetic operations 
(addition, subtraction, multiplication, and division) are supported for CPU and CUDA. 
For samples of usage, see `tests/test_verify_mpint.cu` and `tests/test_perf_mpint.cu`.

GRNS can be freely used in various software-based RNS applications, e.g. RSA, Diffie-Hellman, and Elliptic curves. 
Currently, the algorithms implemented in GRNS are used in a multiple-precision GPU accelerated BLAS library; 
see https://github.com/kisupov/mpres-blas.

At the initialization phase, GRNS relies on two open source libraries.
These are the GNU Multiple Precision Library (GMP) and the GNU MPFR Library (MPFR).
For the division benchmark, the CAMPARY library is additionally required.

### References

1. K. Isupov, "Using floating-point intervals for non-modular computations in residue number system," IEEE Access, vol. 8, pp. 58603-58619, 2020, doi 10.1109/ACCESS.2020.2982365.


*Link: http://github.com/kisupov/grns*

*Copyright 2019, 2020 by Konstantin Isupov.*
