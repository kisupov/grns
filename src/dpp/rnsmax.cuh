/*
 *  Data-parallel primitive
 *  Calculation of the maximum element of an array of RNS numbers using floating-point interval evaluations
 */

#ifndef GRNS_MAX_CUH
#define GRNS_MAX_CUH

#include "rnscmp.cuh"

namespace cuda {

    /*!
     * Todo: try to reduce the number of conditions in this function
     * This method compares two RNS numbers (elements of the array of RNS numbers) ranging from 0 to M-1 using their preprocessed interval estimates.
     * @param ex - interval evaluation of x
     * @param ey - interval evaluation of y
     * @param array - array of RNS numbers
     * @return:
     *  0, if x = y
     *  1, if x > y
     * -1, if x < y
     */
    DEVICE_CUDA_FORCEINLINE int rns_max_cmp(xinterval_t *ex, xinterval_t *ey, int *array) {
        if(ey->val < 0){
            return 1;
        }
        if(ex->val < 0){
            return -1;
        }
        if(cuda::er_ucmp(&ex->low, &ey->upp) > 0){
            return 1;
        }
        if(cuda::er_ucmp(&ey->low, &ex->upp) > 0){
            return -1;
        }
        bool equals = true;
        int idx = ex->val * RNS_MODULI_SIZE;
        int idy = ey->val * RNS_MODULI_SIZE;
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            if(array[idx + i] != array[idy + i]){
                equals = false;
                break;
            }
        }
        return equals ? 0 : cuda::mrc_compare_rns(&array[idx], &array[idy]);
    }

    /*!
      * Kernel that calculates interval evaluations for an array of RNS numbers. Each interval evaluation is computed by
      * a single thread, and multiple interval evaluations are computed at once.
      * @param out - pointer to the computed interval evaluations. Each instance contains the index of the corresponding RNS number
      * @param in - pointer to the array of RNS numbers, size at least N * RNS_MODULI_SIZE
      * @param N - number of elements in the array
      */
    __global__ void rns_max_eval_kernel(xinterval_t *out, int *in, unsigned int N){
        auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        while (numberIdx < N){
            cuda::rns_eval_compute(&out[numberIdx].low, &out[numberIdx].upp, &in[numberIdx * RNS_MODULI_SIZE]);
            out[numberIdx].val = numberIdx;
            numberIdx +=  gridDim.x * blockDim.x;
        }
    }

    /*!
     * Kernel that builds a reduction tree to compute the maximum element of an array of RNS numbers. Each thread block
     * finds the maximum element in its chunk of the input array. In general, instead of the array of RNS numbers (in_num),
     * the array of interval evaluations (in_eval) is used, so that in_num is accessed only in ambiguous cases.
     * This kernel requires blockDim.x * sizeof(xinterval_t) of shared memory
     * @param out - pointer to the interval evaluations of the maximum element in the array of RNS numbers for each thread block. The result
     * of ith block is stored in out[i].
     * @param in_eval - pointer to the input array of interval evaluations
     * @param in_num - pointer to the input array of RNS numbers, size at least N * RNS_MODULI_SIZE
     * @param N - number of elements in the array
     */
    __global__ void rns_max_tree_kernel(xinterval_t *out, xinterval_t *in_eval, int *in_num, unsigned int N){
        auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        extern __shared__ xinterval_t sh_eval[];
        sh_eval[threadIdx.x].val = -1;
        // reduce multiple elements per thread
        while (numberIdx < N){
            if(cuda::rns_max_cmp(&in_eval[numberIdx], &sh_eval[threadIdx.x], in_num) == 1){
                sh_eval[threadIdx.x] = in_eval[numberIdx];
            }
            numberIdx +=  gridDim.x * blockDim.x;
        }
        __syncthreads();
        // do reduction in shared mem
        auto i = cuda::NEXT_POW2(blockDim.x) >> 1; //least power of two greater than or equal to blockDim.x
        while(i >= 1) {
            if ((threadIdx.x < i)
            && (threadIdx.x + i < blockDim.x)
            && cuda::rns_max_cmp(&sh_eval[threadIdx.x + i], &sh_eval[threadIdx.x], in_num) == 1) {
                sh_eval[threadIdx.x] = sh_eval[threadIdx.x + i];
            }
            i = i >> 1;
            __syncthreads();
        }
        //Writing the index of the maximum element for this block to global memory
        if (threadIdx.x == 0) out[blockIdx.x] = sh_eval[threadIdx.x];
    }

    /*
     * Kernel that assigns the value of an RNS number from the 'in' array to out.
     * The index of the element in the array is computed as eval.val.
     * This kernel should be run with one thread block of RNS_MODULI_SIZE threads
     */
    __global__ void rns_max_set_kernel(int *out, int *in, xinterval_t *eval){
        int idx = eval[0].val * RNS_MODULI_SIZE + threadIdx.x;
        out[threadIdx.x] = in[idx];
    }


    /*!
     * Computes the maximum element of an array of RNS numbers using interval floating-point evaluations
     * To reduce across a complete grid (many thread blocks), the computation is split into two kernel launches.
     * The first kernel generates and stores partial reduction results, and the second kernel reduces the partial results into a single total.
     * When looking for the maximum element, we don't really need to write the RNS numbers (partial results) into memory.
     * Instead, we store the interval evaluation and index of the maximum element using the xinterval_t data type
     * @tparam gridDim1  - number of thread blocks to launch the kernel that calculates interval evaluations
     * @tparam blockDim1 - number of threads per block to launch the kernel that calculates interval evaluations
     * @tparam gridDim2 -  number of thread blocks to launch the kernel that perform parallel reduction
     * @tparam blockDim2 - number of threads per block to launch the kernel that perform parallel reduction
     * @param out - pointer to the maximum RNS number in the global GPU memory, size = RNS_MODULI_SIZE
     * @param in - pointer to the array of RNS numbers of the form [x1,x2,...,xn][x1,x2,...,xn][x1,x2,...,xn]...[x1,x2,...,xn], size at least N * RNS_MODULI_SIZE
     * @param N - number of elements in the array
     * @param buffer - global memory buffer of size at least N
     */
    template <int gridDim1, int blockDim1, int gridDim2, int blockDim2>
    void rns_max(int *out, int *in, unsigned int N, xinterval_t *buffer) {

        size_t memSize = blockDim2 * sizeof(xinterval_t);

        cuda::rns_max_eval_kernel <<< gridDim1, blockDim1 >>> ( buffer, in, N);

        cuda::rns_max_tree_kernel <<< gridDim2, blockDim2, memSize >>> (buffer, buffer, in, N);

        cuda::rns_max_tree_kernel <<< 1, blockDim2, memSize >>> (buffer, buffer, in, gridDim2);

        cuda::rns_max_set_kernel <<< 1, RNS_MODULI_SIZE >>> (out, in, buffer);

        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }

}

#endif  //GRNS_MAX_CUH