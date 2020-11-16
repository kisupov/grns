/*
 *  Data generation and conversion functions
 */

#ifndef GRNS_TEST_TSTHELPER_CUH
#define GRNS_TEST_TSTHELPER_CUH

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <random>
#include <chrono>
#include "mpfr.h"
#include "../src/params.h"

/*
 * Bound type for pseudo random number generation
 */
enum RandomBoundType{
    BND_RNS_MODULI_PRODUCT, // up to M-1
    BND_RNS_MODULI_PRODUCT_HALF, // up to M-1
    BND_RNS_MODULI_PRODUCT_SQRT // up to sqrt(M)
};

/*
 * Filling a linear array of RNS numbers with random numbers
 * The array is represented as follows: [x1,x2,...,xn][x1,x2,...,xn]..[x1,x2,...,xn]
 * @param array - pointer to the array of RNS numbers (preliminary memory allocation is required)
 * @param n - size of array to be filled
 * @param boundType - type of random bound
 */
void fill_random_array(int *array, size_t n, RandomBoundType boundType) {
    mpz_t randNum;                                  // Hold our random numbers
    mpz_t rndBnd;                                   // Bound for mpz_urandomm
    gmp_randstate_t state;                          // Random generator state object
    gmp_randinit_default(state);                    // Initialize state for a Mersenne Twister algorithm
    gmp_randseed_ui(state, (unsigned) time(NULL));   // Call gmp_randseed_ui to set initial seed value into state
    mpz_init(randNum);
    mpz_init(rndBnd);
    switch (boundType){
        case BND_RNS_MODULI_PRODUCT:
            mpz_sub_ui(rndBnd, RNS_MODULI_PRODUCT, 1);
            break;
        case BND_RNS_MODULI_PRODUCT_SQRT:
            mpz_sqrt(rndBnd, RNS_MODULI_PRODUCT);
            break;
        default:
            mpz_sub_ui(rndBnd, RNS_MODULI_PRODUCT, 1);
            break;
    }
    //mpz_set_ui(rndBnd, 10);
    for (auto i = 0; i < n; i++) {
        mpz_urandomm(randNum, state, rndBnd);
        rns_from_binary(&array[i*RNS_MODULI_SIZE], randNum);
    }
    gmp_randclear(state);
    mpz_clear(randNum);
    mpz_clear(rndBnd);
}

/*
 * Filling an array of GMP (mpz_t) numbers with random numbers
 * @param array - pointer to the 1D array of mpz_t numbers (preliminary memory allocation is required)
 * @param n - size of array to be filled
 * @param boundType - type of random bound
 * @param allowNegative - true if negative values are permitted
 */
void fill_random_array(mpz_t *array, size_t n, RandomBoundType boundType, bool allowNegative) {
    mpz_t rndBnd;                                   // Bound for mpz_urandomm
    gmp_randstate_t state;                          // Random generator state object
    gmp_randinit_default(state);                    // Initialize state for a Mersenne Twister algorithm
    gmp_randseed_ui(state, (unsigned) time(NULL));   // Call gmp_randseed_ui to set initial seed value into state
    mpz_init(rndBnd);
    mpz_t temp;
    mpz_init(temp);
    switch (boundType){
        case BND_RNS_MODULI_PRODUCT:
            mpz_sub_ui(rndBnd, RNS_MODULI_PRODUCT, 1);
            break;
        case BND_RNS_MODULI_PRODUCT_SQRT:
            mpz_sqrt(rndBnd, RNS_MODULI_PRODUCT);
            break;
        case BND_RNS_MODULI_PRODUCT_HALF:
            mpz_sub_ui(rndBnd, RNS_MODULI_PRODUCT, 1);
            mpz_div_ui(rndBnd, rndBnd,2);
            break;
        default:
            mpz_sub_ui(rndBnd, RNS_MODULI_PRODUCT, 1);
            break;
    }
    for (auto i = 0; i < n; i++) {
        mpz_urandomm(array[i], state, rndBnd);
        if(allowNegative){
            mpz_mod_ui(temp, array[i], 2);
            if (mpz_cmp_ui(temp, 0) == 0){
                mpz_mul_si(array[i], array[i], -1);
            }
        }
    }
    gmp_randclear(state);
    mpz_clear(rndBnd);
    mpz_clear(temp);
}

/*
 * Filling an array of integers with random numbers
 * @param array - pointer to the array of RNS numbers (preliminary memory allocation is required)
 * @param n - size of array to be filled
 */
void fill_random_array(int *array, size_t n) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0);
    for(auto i = 0; i < n; i++){
        array[i] = dist(mt);
    }
}

#endif //GRNS_TEST_TSTHELPER_CUH
