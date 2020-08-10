/*
 *  Basic methods and constants for calculations in the RNS
 */

#ifndef GRNS_BASE_CUH
#define GRNS_BASE_CUH

#include "gmp.h"
#include "mpfr.h"
#include "extrange.cuh"
#include "modular.cuh"

/********************* Precomputed data *********************/

mpz_t  RNS_MODULI_PRODUCT; // Product of all RNS moduli, M = m_1 * m_2 * ... * m_n
mpfr_t RNS_MODULI_PRODUCT_MPFR; // Product of all RNS moduli in the MPFR type, M = m_1 * m_2 * ... * m_n
mpz_t  RNS_PART_MODULI_PRODUCT[RNS_MODULI_SIZE]; // Partial products of moduli, M_i = M / m_i (i = 1,...,n)
int    RNS_PART_MODULI_PRODUCT_INVERSE[RNS_MODULI_SIZE]; // Modulo m_i multiplicative inverses of M_i (i = 1,...,n)
mpz_t  RNS_ORTHOGONAL_BASE[RNS_MODULI_SIZE]; // Orthogonal bases of the RNS, B_i = M_i * RNS_PART_MODULI_PRODUCT_INVERSE[i] (i = 1,...,n)
int    RNS_ONE[RNS_MODULI_SIZE]; // 1 in the RNS
double RNS_MODULI_RECIPROCAL[RNS_MODULI_SIZE]; // Array of 1 / RNS_MODULI[i]
/*
 * Residue codes of 2^j (j = 0,....,RNS_MODULI_PRODUCT_LOG2).
 * Each j-th row contains ( 2^j mod m_1, 2^j mod m_2, 2^j mod m_3 ... )
 * Memory addressing is as follows:
 *      RNS_POW2[0] -> residue code of 2^0
 *      RNS_POW2[1] -> residue code of 2^1
 *      RNS_POW2[RNS_MODULI_PRODUCT_LOG2] -> residue code of 2^RNS_MODULI_PRODUCT_LOG2
 */
int RNS_POW2[RNS_MODULI_PRODUCT_LOG2+1][RNS_MODULI_SIZE];

/*
 * Residues of the RNS moduli product (M) modulo 2^j (j = 1,...,RNS_P2_SCALING_THRESHOLD)
 * This constant is used to power-of-two RNS scaling.
 * Memory addressing is as follows:
 *      RNS_MODULI_PRODUCT_POW2_RESIDUES[0] -> M mod 2^1
 *      RNS_MODULI_PRODUCT_POW2_RESIDUES[1] -> M mod 2^2
 *      RNS_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD-1] -> M mod 2^RNS_P2_SCALING_THRESHOLD
 */
int RNS_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD];

/*
 * Matrix of residues of the partial RNS moduli products (M_i) modulo 2^j (j = 1,...,RNS_P2_SCALING_THRESHOLD)
 * Each j-th row contains ( M_1 mod 2^j, M_2 mod 2^j, M_3 mod 2^j ... )
 * This constant is used to power-of-two RNS scaling.
 * Memory addressing is as follows:
 *      RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[0] -> M_i mod 2^1
 *      RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[1] -> M_i mod 2^2
 *      RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD-1] -> M_i mod 2^RNS_P2_SCALING_THRESHOLD
 */
int RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];

/*
 * Array of multiplicative inverses of 2^1, 2^2, ..., 2^RNS_P2_SCALING_THRESHOLD modulo m_i (i = 1,...,n)
 * Each j-th row contains ( (2^j)^-1 mod m_1, (2^j)^-1 mod m_2, (2^j)^-1 mod m_3 ... )
 * This constant is used to power-of-two RNS scaling.
 * Memory addressing is as follows:
 *      RNS_POW2_INVERSE[0] -> 2^-1 mod m_i
 *      RNS_POW2_INVERSE[1] -> 4^-1 mod m_i
 *      RNS_POW2_INVERSE[RNS_P2_SCALING_THRESHOLD-1] -> (2^RNS_P2_SCALING_THRESHOLD)^{-1} mod m_i
 */
int RNS_POW2_INVERSE[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];

/*
 * Constants for computing the interval evaluation of an RNS number
 */
double RNS_EVAL_ACCURACY; // Accuracy constant for computing the RNS interval evaluation. RNS_EVAL_ACCURACY = 4*u*n*log_2(n)*(1+RNS_EVAL_RELATIVE_ERROR/2)/RNS_EVAL_RELATIVE_ERROR, where u is the unit roundoff,
int RNS_EVAL_REF_FACTOR; // Refinement factor for computing the RNS interval evaluation
int RNS_EVAL_POW2_REF_FACTOR[RNS_MODULI_SIZE]; //The RNS representation of 2^RNS_EVAL_REFINEMENT_FACTOR, (2^RNS_EVAL_REFINEMENT_FACTOR mod m1,..., 2^RNS_EVAL_REFINEMENT_FACTOR mod mn), which is used in the refinement loop
interval_t RNS_EVAL_UNIT; // Interval approximation of 1 / M
interval_t RNS_EVAL_INV_UNIT; // Interval approximation of (M - 1) / M
er_float_t RNS_EVAL_ZERO_BOUND = (er_float_t) {0.0, 0}; // To set the zero interval evaluation

/*
 * Mixed-radix conversion (MRC) constants
 */
int MRC_MULT_INV[RNS_MODULI_SIZE][RNS_MODULI_SIZE]; // Triangle matrix with elements | mult.inv(m_i) |m_j
mpz_t MRS_BASES[RNS_MODULI_SIZE]; // Mixed-radix system bases (1, m1, m1m2, ...., m1m2..m(n-1) )

/*
 * Constants for GPU
 */
namespace cuda {
    __device__ __constant__ int RNS_PART_MODULI_PRODUCT_INVERSE[RNS_MODULI_SIZE];
    __device__ int RNS_POW2[RNS_MODULI_PRODUCT_LOG2+1][RNS_MODULI_SIZE];
    __device__ int RNS_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD];
    __device__ int RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];
    __device__ int RNS_POW2_INVERSE[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];
    __device__ __constant__ double RNS_EVAL_ACCURACY;
    __device__ __constant__ int RNS_EVAL_REF_FACTOR;
    __device__ int RNS_EVAL_POW2_REF_FACTOR[RNS_MODULI_SIZE];
    __device__ __constant__  interval_t RNS_EVAL_UNIT;
    __device__ __constant__  interval_t RNS_EVAL_INV_UNIT;
    __device__ __constant__ er_float_t RNS_EVAL_ZERO_BOUND;
    __device__ int MRC_MULT_INV[RNS_MODULI_SIZE][RNS_MODULI_SIZE];
}


/********************* Helper functions *********************/


/*
 * Computes 2^p
 */
static unsigned int pow2i(int p) {
    unsigned int pow = 1;
    for (int i = 0; i < p; i++)
        pow = (pow * 2);
    return pow;
}

/*
 * Finds modulo m inverse of x using non-recursive extended Euclidean Algorithm
 */
static int mod_inverse(long x, long m){
    long a = x;
    long b = m;
    long result = 0;
    long temp1 = 1;
    long temp2 = 0;
    long temp3 = 0;
    long temp4 = 1;
    long q;
    while(1) {
        q = a / b;
        a = a % b;
        temp1 = temp1 - q * temp2;
        temp3 = temp3 - q * temp4;
        if (a == 0) {
            result = temp2;
            break;
        }
        q = b / a;
        b = b % a;
        temp2 = temp2 - q * temp1;
        temp4 = temp4 - q * temp3;
        if (b == 0) {
            result = temp1;
            break;
        }
    }
    result = (result % m + m) % m;
    return (int) result;
}

/*
 *  Finds modulo m inverse of x using non-recursive extended Euclidean Algorithm,
 *  multiple-precision version
 */
static int mod_inverse_mpz(mpz_t x, int m) {
    mpz_t a;
    mpz_init(a);
    mpz_set(a, x);

    mpz_t b;
    mpz_init(b);
    mpz_set_si(b, m);

    mpz_t result;
    mpz_init(result);
    mpz_set_ui(result, 0);

    mpz_t temp0;
    mpz_init(temp0);

    mpz_t temp1;
    mpz_init(temp1);
    mpz_set_ui(temp1, 1);

    mpz_t temp2;
    mpz_init(temp2);
    mpz_set_ui(temp2, 0);

    mpz_t temp3;
    mpz_init(temp3);
    mpz_set_ui(temp3, 0);

    mpz_t temp4;
    mpz_init(temp4);
    mpz_set_ui(temp4, 1);

    mpz_t q;
    mpz_init(q);

    while(1) {
        mpz_fdiv_q(q, a, b);
        mpz_mod(a, a, b);

        // temp1 = temp1 - q * temp2
        mpz_mul(temp0, q, temp2);
        mpz_sub(temp1, temp1, temp0);

        //temp3 = temp3 - q * temp4;
        mpz_mul(temp0, q, temp4);
        mpz_sub(temp3, temp3, temp0);

        if (mpz_cmp_ui(a, 0) == 0){
            mpz_set(result, temp2);
            break;
        }
        mpz_fdiv_q(q, b, a);
        mpz_mod(b, b, a);

        //temp2 = temp2 - q * temp1;
        mpz_mul(temp0, q, temp1);
        mpz_sub(temp2, temp2, temp0);

        //temp4 = temp4 - q * temp3;
        mpz_mul(temp0, q, temp3);
        mpz_sub(temp4, temp4, temp0);

        if (mpz_cmp_ui(b, 0) == 0){
            mpz_set(result, temp1);
            break;
        }
    }
    mpz_mod_ui(result, result, m);
    mpz_add_ui(result, result, m);
    mpz_mod_ui(result, result, m);
    long inverse = mpz_get_ui(result);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(result);
    mpz_clear(temp0);
    mpz_clear(temp1);
    mpz_clear(temp2);
    mpz_clear(temp3);
    mpz_clear(temp4);
    mpz_clear(q);
    return (int) inverse;
}

/*
 * Set RNS number target from another RNS number x
 */
inline void rns_set(int *target, int * x) {
    memcpy(target, x, sizeof(int) * RNS_MODULI_SIZE);
}

/*
 * Converts x from binary system to RNS
 * The result is stored in target
 */
GCC_FORCEINLINE void rns_from_binary(int * target, mpz_t x) {
    mpz_t residue;
    mpz_init(residue);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mod_ui(residue, x, RNS_MODULI[i]);
        target[i] = mpz_get_ui(residue);
    }
    mpz_clear(residue);
}

/*
 * Converts a two's complement integer x to RNS
 * The result is stored in target
 */
GCC_FORCEINLINE void rns_from_int(int *target, int x) {
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        target[i] = x % RNS_MODULI[i];
    }
}

/*
 * Converts x from RNS to binary system using Chinese remainder theorem.
 * The result is stored in target
 */
GCC_FORCEINLINE void rns_to_binary(mpz_t target, int * x) {
    mpz_t term;
    mpz_init(term);
    mpz_set_ui(target, 0);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_ui(term, RNS_ORTHOGONAL_BASE[i], x[i]);
        mpz_add(target, target, term);
    }
    mpz_mod(target, target, RNS_MODULI_PRODUCT);
    mpz_clear(term);
}

/*
 * Converts x from RNS to the double precision format.
 */
GCC_FORCEINLINE double rns_to_double(int * x) {
    mpz_t m;
    mpz_init(m);
    rns_to_binary(m, x);
    double d = mpz_get_d(m);
    mpz_clear(m);
    return d;
}

/*
 * Computing the EXACT fractional representation of x (i.e. x/M) using CRT
 */
void rns_fractional(er_float_ptr result, int * x) {
    mpfr_t mpfr;
    mpfr_init(mpfr);
    mpz_t m;
    mpz_init(m);
    rns_to_binary(m, x);
    mpfr_set_z(mpfr, m, MPFR_RNDN);
    mpfr_div(mpfr, mpfr, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDN);
    result->frac = mpfr_get_d_2exp((long *) &result->exp, mpfr, MPFR_RNDN);
    mpfr_clear(mpfr);
    mpz_clear(m);
}


/********************* Functions for calculating and printing the RNS constants *********************/


/*
 * Init precomputed RNS data
 */
void rns_const_init(){
    mpz_init(RNS_MODULI_PRODUCT);
    mpfr_init(RNS_MODULI_PRODUCT_MPFR);
    mpz_set_ui(RNS_MODULI_PRODUCT, 1);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_init(RNS_PART_MODULI_PRODUCT[i]);
        mpz_init(RNS_ORTHOGONAL_BASE[i]);
    }
    //Computing moduli product, M
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_ui(RNS_MODULI_PRODUCT, RNS_MODULI_PRODUCT, RNS_MODULI[i]);
    }
    mpfr_set_z(RNS_MODULI_PRODUCT_MPFR, RNS_MODULI_PRODUCT, MPFR_RNDD);

    //Computing partial products, M_i = M / m_i
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_div_ui(RNS_PART_MODULI_PRODUCT[i], RNS_MODULI_PRODUCT, RNS_MODULI[i]);
    }
    //Computing multiplicative inverse of M_i modulo m_i
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        RNS_PART_MODULI_PRODUCT_INVERSE[i] = mod_inverse_mpz(RNS_PART_MODULI_PRODUCT[i], RNS_MODULI[i]);

    }
    //Computing orthogonal bases
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_si(RNS_ORTHOGONAL_BASE[i], RNS_PART_MODULI_PRODUCT[i], RNS_PART_MODULI_PRODUCT_INVERSE[i]);
    }
    //Computing reciprocals for moduli
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        RNS_MODULI_RECIPROCAL[i] = (double) 1 / RNS_MODULI[i];
    }
    //Setting 1 in the RNS
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        RNS_ONE[i] = 1;
    }
    //Computing RNS representations for powers of two
    for (int i = 0; i <= RNS_MODULI_PRODUCT_LOG2; i++) {
        for (int j = 0; j < RNS_MODULI_SIZE; j++) {
            RNS_POW2[i][j] = 1;
            for (int k = 0; k < i; k++)
                RNS_POW2[i][j] = mod_mul(RNS_POW2[i][j], 2, RNS_MODULI[j]); //(RNS_POW2[i][j] * 2) % RNS_MODULI[j];
        }
    }
    //Computing the residues of moduli product (M) modulo 2^j (j = 1,...,RNS_P2_SCALING_THRESHOLD)
    mpz_t residue;
    mpz_init(residue);
    for (int i = 0; i < RNS_P2_SCALING_THRESHOLD; i++) {
        mpz_mod_ui(residue, RNS_MODULI_PRODUCT, pow2i(i+1));
        RNS_MODULI_PRODUCT_POW2_RESIDUES[i] = (int)mpz_get_si(residue);
    }
    //Computing the matrix of residues of the partial RNS moduli products (M_i) modulo 2^j (the H matrix)
    for (int i = 0; i < RNS_P2_SCALING_THRESHOLD; i++) {
        for(int j = 0; j < RNS_MODULI_SIZE; j++){
            mpz_mod_ui(residue, RNS_PART_MODULI_PRODUCT[j], pow2i(i+1));
            RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[i][j] = (int)mpz_get_si(residue);
        }
    }
    //Computing multiplicative inverses of 2^1,2^2,...,2^RNS_P2_SCALING_THRESHOLD modulo m_i
    for (int i = 0; i < RNS_P2_SCALING_THRESHOLD; i++) {
        for (int j = 0; j < RNS_MODULI_SIZE; j++) {
            int pow = 1;
            for (int k = 0; k <= i; k++){
                pow = mod_mul(pow, 2, RNS_MODULI[j]);     //(pow * 2) % RNS_MODULI[j];
            }
            RNS_POW2_INVERSE[i][j] = mod_inverse((long)pow, (long)RNS_MODULI[j]);
        }
    }
    mpz_clear(residue);
    //Computing accuracy constant for RNS interval evaluation
    RNS_EVAL_ACCURACY  =  4 * pow(2.0, 1 - DBL_PRECISION) * RNS_MODULI_SIZE * log2((double)RNS_MODULI_SIZE) * (1 + RNS_EVAL_RELATIVE_ERROR / 2)  / RNS_EVAL_RELATIVE_ERROR;
    // Computing refinement coefficient for RNS interval evaluation
    RNS_EVAL_REF_FACTOR  =  floor(log2(1/(2*RNS_EVAL_ACCURACY)));
    // Computing two in degree of RNS_EVAL_REF_FACTOR in the RNS representation
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        RNS_EVAL_POW2_REF_FACTOR[i] = 1;
        for (int k = 0; k < RNS_EVAL_REF_FACTOR; k++){
            RNS_EVAL_POW2_REF_FACTOR[i] = mod_mul(RNS_EVAL_POW2_REF_FACTOR[i], 2, RNS_MODULI[i]);
        }
    }
    mpfr_t mpfr_tmp, mpfr_one;
    mpfr_init2(mpfr_tmp, 10000);
    mpfr_init2(mpfr_one, 10000);
    mpfr_set_ui(mpfr_one, 1, MPFR_RNDN);
    //Computing upper bound for 1 / M
    mpfr_set_ui(mpfr_tmp, 1, MPFR_RNDN);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDU);
    RNS_EVAL_UNIT.upp.frac = mpfr_get_d_2exp(&RNS_EVAL_UNIT.upp.exp, mpfr_tmp, MPFR_RNDU);
    //Computing upper bound for (M - 1) / M = 1 - 1 / M
    mpfr_sub(mpfr_tmp, mpfr_one, mpfr_tmp, MPFR_RNDU);
    RNS_EVAL_INV_UNIT.upp.frac = mpfr_get_d_2exp(&RNS_EVAL_INV_UNIT.upp.exp, mpfr_tmp, MPFR_RNDU);
    //Computing lower bound for 1 / M
    mpfr_set_ui(mpfr_tmp, 1, MPFR_RNDN);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    RNS_EVAL_UNIT.low.frac = mpfr_get_d_2exp(&RNS_EVAL_UNIT.low.exp, mpfr_tmp, MPFR_RNDD);
    //Computing lower bound for (M - 1) / M = 1 - 1 / M
    mpfr_sub(mpfr_tmp, mpfr_one, mpfr_tmp, MPFR_RNDD);
    RNS_EVAL_INV_UNIT.low.frac = mpfr_get_d_2exp(&RNS_EVAL_INV_UNIT.low.exp, mpfr_tmp, MPFR_RNDD);
    mpfr_clear(mpfr_tmp);
    mpfr_clear(mpfr_one);
    //Init the MRC constants
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        for (int j = 0; j < RNS_MODULI_SIZE; ++j) {
            MRC_MULT_INV[i][j] = 0;
        }
    }
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        for (int j = i + 1; j < RNS_MODULI_SIZE; ++j) {
            MRC_MULT_INV[i][j] = mod_inverse((long)RNS_MODULI[i], (long)RNS_MODULI[j]);
        }
    }
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        mpz_init(MRS_BASES[i]);
        mpz_set_ui(MRS_BASES[i], 1);
        for (int j = 0; j < i ; ++j) {
            mpz_mul_ui(MRS_BASES[i], MRS_BASES[i], RNS_MODULI[j]);
        }
    }
    //Copying constants to the GPU memory
    cudaMemcpyToSymbol(cuda::RNS_MODULI, &RNS_MODULI, RNS_MODULI_SIZE * sizeof(int)); // Declared in modular.cuh
    cudaMemcpyToSymbol(cuda::RNS_PART_MODULI_PRODUCT_INVERSE, &RNS_PART_MODULI_PRODUCT_INVERSE, RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_POW2, &RNS_POW2, (RNS_MODULI_PRODUCT_LOG2+1) * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_MODULI_PRODUCT_POW2_RESIDUES, &RNS_MODULI_PRODUCT_POW2_RESIDUES, RNS_P2_SCALING_THRESHOLD * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES, &RNS_PART_MODULI_PRODUCT_POW2_RESIDUES, RNS_P2_SCALING_THRESHOLD * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_POW2_INVERSE, &RNS_POW2_INVERSE, RNS_P2_SCALING_THRESHOLD * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_ACCURACY, &::RNS_EVAL_ACCURACY, sizeof(double));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_REF_FACTOR, &::RNS_EVAL_REF_FACTOR, sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_POW2_REF_FACTOR, &::RNS_EVAL_POW2_REF_FACTOR, sizeof(int) * RNS_MODULI_SIZE);
    cudaMemcpyToSymbol(cuda::RNS_EVAL_UNIT, &RNS_EVAL_UNIT, sizeof(interval_t));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_INV_UNIT, &RNS_EVAL_INV_UNIT, sizeof(interval_t));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_ZERO_BOUND, &RNS_EVAL_ZERO_BOUND, sizeof(er_float_t));
    cudaMemcpyToSymbol(cuda::MRC_MULT_INV, &MRC_MULT_INV, sizeof(int) * RNS_MODULI_SIZE * RNS_MODULI_SIZE);
}

/*
 * Printing the constants of the RNS
 */
void rns_const_print(bool briefly) {
    std::cout << "Constants of the RNS system:" << std::endl;
    std::cout << "- RNS_MODULI_SIZE, n: " << RNS_MODULI_SIZE << std::endl;
    std::cout << "- RNS_MODULI_PRODUCT, M: " << mpz_get_str(NULL, 10, RNS_MODULI_PRODUCT) << std::endl;
    mpfr_t log2;
    mpfr_init(log2);
    mpfr_log2(log2, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    std::cout << "- BIT-SIZE OF MODULI PRODUCT, LOG2(M): " << mpfr_get_d(log2, MPFR_RNDN) << std::endl;
    mpfr_clear(log2);
    std::cout << "- RNS_P2_SCALING_THRESHOLD, T: " << RNS_P2_SCALING_THRESHOLD << std::endl;
    std::cout << "- RNS_PARALLEL_REDUCTION_IDX: " << RNS_PARALLEL_REDUCTION_IDX << std::endl;
    if (!briefly) {
        std::cout << "- RNS_MODULI, m_i: ";
        for (int i = 0; i < RNS_MODULI_SIZE; i++)
            std::cout << std::endl << RNS_MODULI[i];
        std::cout << std::endl;
        std::cout << "- RNS_PART_MODULI_PRODUCT, M_i: ";
        for (int i = 0; i < RNS_MODULI_SIZE; i++)
            std::cout << std::endl << mpz_get_str(NULL, 10, RNS_PART_MODULI_PRODUCT[i]);
        std::cout << std::endl;
        std::cout << "- RNS_PART_MODULI_PRODUCT_INVERSE, (M_i)^-1: ";
        for (int i = 0; i < RNS_MODULI_SIZE; i++)
            std::cout << std::endl << RNS_PART_MODULI_PRODUCT_INVERSE[i];
        std::cout << std::endl;
        std::cout <<  "- RNS_ORTHOGONAL_BASE, M_i * (M_i)^-1: ";
        for (int i = 0; i < RNS_MODULI_SIZE; i++)
            std::cout << std::endl << mpz_get_str(NULL, 10, RNS_ORTHOGONAL_BASE[i]);
        std::cout << std::endl;
    }
}


#endif //GRNS_BASE_CUH
