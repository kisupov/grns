# Library for computations in Residue Number System (RNS) using CUDA-enabled GPUs (GRNS)
###### Version 0.1

The following time-consuming (aka non-modular) operations in RNS are implemented in addition to simple addition, subtraction, and multiplication:

* magnitude comparison (`/src/rnscmp.cuh`)
* general integer division (`/src/rnsdiv.cuh`)
* power-of-two scaling (`/src/rnsscal.cuh`)

To perform these operations, the interval evaluation of the fractional representation of an RNS number is computed (`/src/rnseval.cuh`).
Both serial (single-threaded) and parallel (n-threaded) implementations of the above operations are available.

GRNS is designed for large RNS dynamic ranges, which significantly exceed the usual precision of computers 
(say hundreds and thousands of bits). There are no special restrictions on the moduli sets and 
the magnitude of numbers in RNS representation. Some predefined moduli sets are located in `/src/params`.

GRNS can be freely used in various software-based RNS applications, e.g. RSA, Diffie-Hellman, and elliptic curve cryptography. 
Currently, the algorithms implemented in GRNS are used in a multiple-precision GPU accelerated BLAS library; 
see https://github.com/kisupov/mpres-blas.

At the initialization phase, GRNS relies on two open source libraries.
These are the GNU Multiple Precision Library (GMP) and the GNU MPFR Library (MPFR).
For the division benchmark, the CAMPARY library is additionally required.

For samples of usage, see `/tests`.


*Link: http://github.com/kisupov/grns*

*Copyright 2019, 2020 by Konstantin Isupov.*
