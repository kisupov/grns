/*
 *  Test for checking the correctness of the multiple-precision integer routines
 */


#include "../src/mpint.cuh"
#include "logger.cuh"
#include <iostream>

enum mpint_test_type {
    add_test,
    sub_test,
    mul_test,
    div_test
};

/* Euclidean division. The code is from
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/divmodnote.pdf
 * */
static int divE( int D, int d )
{
    int q = D/d;
    int r = D%d;
    if (r < 0) {
        if (d > 0) q = q-1;
        else q = q+1;
    }
    return q;
}

static void printResult(const  char * name, mp_int_ptr result){
    double d = mpint_get_double(result);
    printf("\n%s = %.8f \t\t", name, d);
}

/*
 * GPU tests
 */

static __global__ void testCudaAdd(mp_int_ptr dz, mp_int_ptr dx, mp_int_ptr dy){
    cuda::mpint_add(dz, dx, dy);
}

static __global__ void testCudaSub(mp_int_ptr dz, mp_int_ptr dx, mp_int_ptr dy){
    cuda::mpint_sub(dz, dx, dy);
}

static __global__ void testCudaMul(mp_int_ptr dz, mp_int_ptr dx, mp_int_ptr dy){
    cuda::mpint_mul(dz, dx, dy);
}

static __global__ void testCudaDiv(mp_int_ptr dz, mp_int_ptr dx, mp_int_ptr dy){
    cuda::mpint_div(dz, dx, dy);
}


void testCuda(mp_int_ptr x, mp_int_ptr y, mp_int_ptr z, mpint_test_type type){
    mp_int_ptr dx;
    mp_int_ptr dy;
    mp_int_ptr dz;
    cudaMalloc(&dx, sizeof(mp_int_t));
    cudaMalloc(&dy, sizeof(mp_int_t));
    cudaMalloc(&dz, sizeof(mp_int_t));
    cudaMemcpy(dx, x, sizeof(mp_int_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(mp_int_t), cudaMemcpyHostToDevice);

    switch(type){
        case add_test:
            testCudaAdd<<<1,1>>>(dz, dx, dy);
            break;
        case sub_test:
            testCudaSub<<<1,1>>>(dz, dx, dy);
            break;
        case mul_test:
            testCudaMul<<<1,1>>>(dz, dx, dy);
            break;
        case div_test:
            testCudaDiv<<<1,1>>>(dz, dx, dy);
            break;
        default:
            break;
    }
    cudaMemcpy(z, dz, sizeof(mp_int_t), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
}


void base_test() {

    int arg_x = -55;
    int arg_y = 13;

    mp_int_t x;
    mp_int_t y;
    mp_int_t z;

    mpint_set_i(&x, arg_x);
    mpint_set_i(&y, arg_y);

    Logger::printDash();
    printf("x = %i\n", arg_x);
    printf("y = %i\n", arg_y);

    Logger::printDash();
    printf("EXACT        = %i \t\t", arg_x + arg_y);
    mpint_add(&z, &x, &y);
    printResult("[CPU]  x + y", &z);
    mpint_set_i(&z, 0);
    testCuda(&x, &y, & z,add_test);
    printResult("[CUDA] x + y", &z);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("EXACT        = %i \t\t", arg_x - arg_y);
    mpint_sub(&z, &x, &y);
    printResult("[CPU]  x - y", &z);
    mpint_set_i(&z, 0);
    testCuda(&x, &y, & z,sub_test);
    printResult("[CUDA] x - y", &z);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("EXACT        = %i \t\t", arg_x * arg_y);
    mpint_mul(&z, &x, &y);
    printResult("[CPU]  x * y", &z);
    mpint_set_i(&z, 0);
    testCuda(&x, &y, & z,mul_test);
    printResult("[CUDA] x * y", &z);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("EXACT        = %i \t\t", divE(arg_x, arg_y));
    mpint_div(&z, &x, &y);
    printResult("[CPU]  x / y", &z);
    mpint_set_i(&z, 0);
    testCuda(&x, &y, & z,div_test);
    printResult("[CUDA] x / y", &z);
    //--------------------------------------------------------------
    Logger::printSpace();
}

int main() {
    Logger::beginTestDescription(Logger::TEST_VERIFY_MPINT);
    Logger::printSpace();
    rns_const_init();
    mpint_const_init();
    rns_const_print(true);
    base_test();
    Logger::printSpace();
    Logger::endTestDescription();
    return 0;
}