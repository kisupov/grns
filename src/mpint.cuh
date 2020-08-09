/*
 *  Multiple-precision (MP) integer arithmetic using Residue number system
 */

#ifndef GRNS_MPINT_CUH
#define GRNS_MPINT_CUH

#include <iostream>
#include "roundings.cuh"
#include "rnsdiv.cuh"

/********************* Global precomputed constants *********************/

mp_int_t MPINT_ZERO; //Zero in the used multiple-precision representation

//Constants for GPU
namespace cuda {
    __device__ __constant__ mp_int_t MPINT_ZERO;
}

void mpint_const_init() {
    //Setting MP_ZERO
    MPINT_ZERO.sign = 0;
    MPINT_ZERO.eval[0].exp = 0;
    MPINT_ZERO.eval[1].exp = 0;
    MPINT_ZERO.eval[0].frac = 0.0;
    MPINT_ZERO.eval[1].frac = 0.0;
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        MPINT_ZERO.digits[i] = 0;
    }
    //Copying constants to the GPU memory
    cudaMemcpyToSymbol(cuda::MPINT_ZERO, &MPINT_ZERO, sizeof(mp_int_t));
}

/********************* Assignment, conversion and formatted output functions *********************/

/*!
 * Set the value of result from x
 */
GCC_FORCEINLINE void mpint_set(mp_int_ptr result, mp_int_ptr x) {
    rns_set(result->digits, x->digits);
    result->sign = x->sign;
    result->eval[0] = x->eval[1];
    result->eval[1] = x->eval[1];
}

/*!
 * Forming a multiple-precision number from the significand and sign
 */
GCC_FORCEINLINE void mpint_set(mp_int_ptr result, long significand, int sign) {
    result->sign = sign;
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        long residue = significand % (long)RNS_MODULI[i];
        result->digits[i] = (int) residue;
    }
    rns_eval_compute(&result->eval[0], &result->eval[1], result->digits);
}

/*!
 *  Set the value of result from x
 */
GCC_FORCEINLINE void mpint_set_i(mp_int_ptr result, int x) {
    if (x >= 0) {
        result->sign = 0;
    } else {
        result->sign = 1;
    }
    rns_from_int(result->digits, abs(x));
    rns_eval_compute(&result->eval[0], &result->eval[1], result->digits);
}

/*!
 *  Set the value of result from x
 */
GCC_FORCEINLINE void mpint_set_mpz(mp_int_ptr result, mpz_t x) {
    if (mpz_cmp_ui(x, 0) >= 0) {
        result->sign = 0;
    } else {
        result->sign = 1;
    }
    rns_from_binary(result->digits, x);
    rns_eval_compute(&result->eval[0], &result->eval[1], result->digits);
}

/*!
 * Convert x to a double
 */
GCC_FORCEINLINE double mpint_get_double(mp_int_ptr x) {
    double d = rns_to_double(x->digits);
    if (x->sign == 1) { d = d * (-1); }
    return d;
}

/*!
 * Convert x to the mpz_t number result
 */
GCC_FORCEINLINE void mpint_get_mpz(mpz_t result, mp_int_ptr x) {
    mpz_t temp;
    mpz_init(temp);
    mpz_set_si(result, 0);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_si(temp, RNS_ORTHOGONAL_BASE[i], x->digits[i]);
        mpz_add(result, result, temp);
    }
    mpz_mod(result, result, RNS_MODULI_PRODUCT);
    if (x->sign == 1) {
        mpz_mul_si(result, result, -1);
    }
    mpz_clear(temp);
}

/*!
 * Print the parts (fields) of a multiple-precision number
 */
void mp_print(mp_int_ptr x) {
    printf("\nMultiple-precision value:\n");
    printf("- Sign: %d\n", x->sign);
    printf("- Significand: <");
    for (int i = 0; i < RNS_MODULI_SIZE; i++)
        printf("%d ", x->digits[i]);
    printf("\b>\n");
    printf("- Eval.low: ");
    er_print(&x->eval[0]);
    printf("\n- Eval.exact: ");
    er_float_t er_temp;
    rns_fractional(&er_temp, x->digits);
    er_print(&er_temp);
    printf("\n- Eval.upp: ");
    er_print(&x->eval[1]);
    printf("\n\n");
}


/********************* Basic multiple-precision arithmetic operations *********************/

/*!
 * Overflow detection using the interval evaluation of the result
 */
GCC_FORCEINLINE void mpint_check_overflow(er_float_ptr lower, er_float_ptr upper, const char *routine){
    if (upper->exp < -1 || er_ucmp(upper, &RNS_EVAL_INV_UNIT.low) <= 0) {
        //If exp = -2 then eval = 1.999999 / 4 < 0.5
        return; //No overflow occurs
    }
    if (er_ucmp(lower, &RNS_EVAL_INV_UNIT.upp) == 1) {
        printf("\nERROR [%s]: Overflow has occurred", routine);
        return;
    }
    printf("\nERROR [%s]: Overflow may occurred (the result needs to be clarified)", routine);
}

/*!
 * Addition of two multiple-precision numbers
 * result = x + y
 * This is a simplified version of the floating-point addition algorithm
 * from the JPDC paper https://www.sciencedirect.com/science/article/pii/S0743731519303302
 */
GCC_FORCEINLINE void mpint_add(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
    int sign_x = x->sign;
    int sign_y = y->sign;
    er_float_t eval_x[2];
    er_float_t eval_y[2];
    eval_x[0] = x->eval[0];
    eval_x[1] = x->eval[1];
    eval_y[0] = y->eval[0];
    eval_y[1] = y->eval[1];

    int alpha = (1 - 2 * sign_x);
    int beta = (1 - 2 * sign_y);

    //Addition of the RNS significands
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        int residue = mod_add(alpha * x->digits[i], beta * y->digits[i], RNS_MODULI[i]);
        result->digits[i] = residue < 0 ? residue + RNS_MODULI[i] : residue;
    }

    //Change the signs of the endpoints of interval evaluation when the number is negative
    //The signs will not change when the number is positive
    eval_x[0].frac *=  alpha;
    eval_x[1].frac *=  alpha;
    eval_y[0].frac *=  beta;
    eval_y[1].frac *=  beta;

    //Interval addition
    er_add_rd(&result->eval[0], &eval_x[sign_x], &eval_y[sign_y]);
    er_add_ru(&result->eval[1], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

    //Restoring the negative result
    //int plus  = result->eval[0].frac >= 0 && result->eval[1].frac >= 0;
    int minus = result->eval[0].frac < 0 && result->eval[1].frac < 0;
    result->sign = minus;
    //One observation (should be proven in the future):
    //when both plus and minus are equal to zero, the actual result is always non-negative.
    if(minus){
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            result->digits[i] = (RNS_MODULI[i] - result->digits[i]) % RNS_MODULI[i];
        }
        er_float_t tmp = result->eval[0];
        result->eval[0].frac = -1 * result->eval[1].frac;
        result->eval[0].exp  = result->eval[1].exp;
        result->eval[1].frac = -1 * tmp.frac;
        result->eval[1].exp  = tmp.exp;
    }
    mpint_check_overflow(&result->eval[0], & result->eval[1], "mpint_add");
}

/*!
 * Subtraction of two multiple-precision numbers
 * result = x - y
 * This is a simplified and slightly modified version of the floating-point addition algorithm
 * from the JPDC paper https://www.sciencedirect.com/science/article/pii/S0743731519303302
 */
GCC_FORCEINLINE void mpint_sub(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
    int sign_x = x->sign;
    int sign_y = y->sign ^ 1;
    er_float_t eval_x[2];
    er_float_t eval_y[2];
    eval_x[0] = x->eval[0];
    eval_x[1] = x->eval[1];
    eval_y[0] = y->eval[0];
    eval_y[1] = y->eval[1];

    int alpha = (1 - 2 * sign_x);
    int beta = (1 - 2 * sign_y);

    //Addition of the RNS significands
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        int residue = mod_add(alpha * x->digits[i], beta * y->digits[i], RNS_MODULI[i]);
        result->digits[i] = residue < 0 ? residue + RNS_MODULI[i] : residue;
    }

    //Change the signs of the endpoints of interval evaluation when the number is negative
    //The signs will not change when the number is positive
    eval_x[0].frac *=  alpha;
    eval_x[1].frac *=  alpha;
    eval_y[0].frac *=  beta;
    eval_y[1].frac *=  beta;

    //Interval addition
    er_add_rd(&result->eval[0], &eval_x[sign_x], &eval_y[sign_y]);
    er_add_ru(&result->eval[1], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

    //Restoring the negative result
    //int plus  = result->eval[0].frac >= 0 && result->eval[1].frac >= 0;
    int minus = result->eval[0].frac < 0 && result->eval[1].frac < 0;
    result->sign = minus;
    //One observation (should be proven in the future):
    //when both plus and minus are equal to zero, the actual result is always non-negative.
    if(minus){
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            result->digits[i] = (RNS_MODULI[i] - result->digits[i]) % RNS_MODULI[i];
        }
        er_float_t tmp = result->eval[0];
        result->eval[0].frac = -1 * result->eval[1].frac;
        result->eval[0].exp  = result->eval[1].exp;
        result->eval[1].frac = -1 * tmp.frac;
        result->eval[1].exp  = tmp.exp;
    }
    mpint_check_overflow(&result->eval[0], & result->eval[1], "mpint_sub");
}

/*!
 * Multiplication of two multiple-precision numbers
 * result = x * y
 * This is a simplified and slightly modified version of the floating-point addition algorithm
 * from the JPDC paper https://www.sciencedirect.com/science/article/pii/S0743731519303302
 */
GCC_FORCEINLINE void mpint_mul(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
    er_md_rd(&result->eval[0], &x->eval[0], &y->eval[0], &RNS_EVAL_UNIT.upp);
    er_md_ru(&result->eval[1], &x->eval[1], &y->eval[1], &RNS_EVAL_UNIT.low);
    for(int i = 0; i < RNS_MODULI_SIZE; i ++){
        result->digits[i] = mod_mul(x->digits[i], y->digits[i], RNS_MODULI[i]);
    }
    result->sign = rns_check_zero(result->digits) ? 0 : x->sign ^ y->sign;
    mpint_check_overflow(&result->eval[0], & result->eval[1], "mpint_mul");
}

/*!
 * Division of two multiple-precision numbers using Euclidean definition
 * result = x / y
 * For RNS division, the algorithm proposed in IEEE Acceess paper is used
 * https://ieeexplore.ieee.org/document/9043511
 */
GCC_FORCEINLINE void mpint_div(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
    int remainder[RNS_MODULI_SIZE];
    rns_div(result->digits, remainder, x->digits, y->digits);
    if(x->sign == 1 && !rns_check_zero(remainder)) {
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            result->digits[i] = (result->digits[i] + 1) % RNS_MODULI[i];
        }
    }
    rns_eval_compute(&result->eval[0], &result->eval[1],result->digits);
    result->sign = rns_check_zero(result->digits) ? 0 : x->sign ^ y->sign;
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Overflow detection using the interval evaluation of the result
     */
    DEVICE_CUDA_FORCEINLINE void mpint_check_overflow(er_float_ptr lower, er_float_ptr upper, const char *routine){
        if (upper->exp < -1 || cuda::er_ucmp(upper, &cuda::RNS_EVAL_INV_UNIT.low) <= 0) {
            //If exp = -2 then eval = 1.999999 / 4 < 0.5
            return; //No overflow occurs
        }
        if (cuda::er_ucmp(lower, &cuda::RNS_EVAL_INV_UNIT.upp) == 1) {
            printf("\nERROR [%s]: Overflow has occurred", routine);
            return;
        }
        printf("\nERROR [%s]: Overflow may occurred (the result needs to be clarified)", routine);
    }

    /*!
     * Addition of two multiple-precision numbers
     * result = x + y
     * This is a simplified version of the floating-point addition algorithm
     * from the JPDC paper https://www.sciencedirect.com/science/article/pii/S0743731519303302
     */
    DEVICE_CUDA_FORCEINLINE void mpint_add(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
        int sign_x = x->sign;
        int sign_y = y->sign;
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x->eval[0];
        eval_x[1] = x->eval[1];
        eval_y[0] = y->eval[0];
        eval_y[1] = y->eval[1];

        int alpha = (1 - 2 * sign_x);
        int beta = (1 - 2 * sign_y);

        //Addition of the RNS significands
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            int residue = cuda::mod_add(alpha * x->digits[i], beta * y->digits[i], cuda::RNS_MODULI[i]);
            result->digits[i] = residue < 0 ? residue + cuda::RNS_MODULI[i] : residue;
        }

        //Change the signs of the endpoints of interval evaluation when the number is negative
        //The signs will not change when the number is positive
        eval_x[0].frac *=  alpha;
        eval_x[1].frac *=  alpha;
        eval_y[0].frac *=  beta;
        eval_y[1].frac *=  beta;

        cuda::er_add_rd(&result->eval[0], &eval_x[sign_x], &eval_y[sign_y]);
        cuda::er_add_ru(&result->eval[1], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

        //Restoring the negative result
        //int plus  = result->eval[0].frac >= 0 && result->eval[1].frac >= 0;
        int minus = result->eval[0].frac < 0 && result->eval[1].frac < 0;
        result->sign = minus;
        //One observation (should be proven in the future):
        //when both plus and minus are equal to zero, the actual result is always non-negative.
        if(minus){
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                result->digits[i] = (cuda::RNS_MODULI[i] - result->digits[i]) % cuda::RNS_MODULI[i];
            }
            er_float_t tmp = result->eval[0];
            result->eval[0].frac = -1 * result->eval[1].frac;
            result->eval[0].exp  = result->eval[1].exp;
            result->eval[1].frac = -1 * tmp.frac;
            result->eval[1].exp  = tmp.exp;
        }
        cuda::mpint_check_overflow(&result->eval[0], & result->eval[1], "mpint_add");
    }

    /*!
     * Subtraction of two multiple-precision numbers
     * result = x - y
     * This is a simplified and slightly modified version of the floating-point addition algorithm
     * from the JPDC paper https://www.sciencedirect.com/science/article/pii/S0743731519303302
     */
    DEVICE_CUDA_FORCEINLINE void mpint_sub(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
        int sign_x = x->sign;
        int sign_y = y->sign ^ 1;
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x->eval[0];
        eval_x[1] = x->eval[1];
        eval_y[0] = y->eval[0];
        eval_y[1] = y->eval[1];

        int alpha = (1 - 2 * sign_x);
        int beta = (1 - 2 * sign_y);

        //Addition of the RNS significands
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            int residue = cuda::mod_add(alpha * x->digits[i], beta * y->digits[i], cuda::RNS_MODULI[i]);
            result->digits[i] = residue < 0 ? residue + cuda::RNS_MODULI[i] : residue;
        }

        //Change the signs of the endpoints of interval evaluation when the number is negative
        //The signs will not change when the number is positive
        eval_x[0].frac *=  alpha;
        eval_x[1].frac *=  alpha;
        eval_y[0].frac *=  beta;
        eval_y[1].frac *=  beta;

        //Interval addition
        cuda::er_add_rd(&result->eval[0], &eval_x[sign_x], &eval_y[sign_y]);
        cuda::er_add_ru(&result->eval[1], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

        //Restoring the negative result
        //int plus  = result->eval[0].frac >= 0 && result->eval[1].frac >= 0;
        int minus = result->eval[0].frac < 0 && result->eval[1].frac < 0;
        result->sign = minus;
        //One observation (should be proven in the future):
        //when both plus and minus are equal to zero, the actual result is always non-negative.
        if(minus){
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                result->digits[i] = (cuda::RNS_MODULI[i] - result->digits[i]) % cuda::RNS_MODULI[i];
            }
            er_float_t tmp = result->eval[0];
            result->eval[0].frac = -1 * result->eval[1].frac;
            result->eval[0].exp  = result->eval[1].exp;
            result->eval[1].frac = -1 * tmp.frac;
            result->eval[1].exp  = tmp.exp;
        }
        cuda::mpint_check_overflow(&result->eval[0], & result->eval[1], "mpint_sub");
    }

    /*!
     * Multiplication of two multiple-precision numbers
     * result = x * y
     * This is a simplified and slightly modified version of the floating-point addition algorithm
     * from the JPDC paper https://www.sciencedirect.com/science/article/pii/S0743731519303302
     */
    DEVICE_CUDA_FORCEINLINE void mpint_mul(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
        cuda::er_md_rd(&result->eval[0], &x->eval[0], &y->eval[0], &cuda::RNS_EVAL_UNIT.upp);
        cuda::er_md_ru(&result->eval[1], &x->eval[1], &y->eval[1], &cuda::RNS_EVAL_UNIT.low);
        for(int i = 0; i < RNS_MODULI_SIZE; i ++){
            result->digits[i] = cuda::mod_mul(x->digits[i], y->digits[i], cuda::RNS_MODULI[i]);
        }
        result->sign = cuda::rns_check_zero(result->digits) ? 0 : x->sign ^ y->sign;
        cuda::mpint_check_overflow(&result->eval[0], & result->eval[1], "mpint_mul");
    }

    /*!
     * Division of two multiple-precision numbers using Euclidean definition
     * result = x / y
     * For RNS division, the algorithm proposed in IEEE Acceess paper is used
     * https://ieeexplore.ieee.org/document/9043511
     */
    DEVICE_CUDA_FORCEINLINE void mpint_div(mp_int_ptr result, mp_int_ptr x, mp_int_ptr y) {
        int remainder[RNS_MODULI_SIZE];
        cuda::rns_div(result->digits, remainder, x->digits, y->digits);
        if(x->sign == 1 && !cuda::rns_check_zero(remainder)) {
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                result->digits[i] = (result->digits[i] + 1) % cuda::RNS_MODULI[i];
            }
        }
        cuda::rns_eval_compute(&result->eval[0], &result->eval[1],result->digits);
        result->sign = cuda::rns_check_zero(result->digits) ? 0 : x->sign ^ y->sign;
    }

} //end of namespace

#endif //GRNS_MPINT_CUH
