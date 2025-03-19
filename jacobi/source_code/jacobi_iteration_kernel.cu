#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(float *A, float *B, float *x, float *new_x, double *ssd, int num_rows, int num_columns)
{
    extern __shared__ double shared_ssd[];

    int blockX = blockIdx.x;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (row < num_rows) {
        double old_value = x[row];

        double sum = 0.0;
        for (int col = 0; col < num_columns; col++) {
            if (row != col) {
                sum += A[row * num_columns + col] * x[col];
            }
        }
        double new_value = (B[row] - sum) / A[row * num_columns + row];

        new_x[row] = new_value;

        double diff = old_value - new_value;
        double local_ssd = diff * diff;

        shared_ssd[tid] = local_ssd;
    }
    else {
        shared_ssd[tid] = 0.0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            shared_ssd[tid] += shared_ssd[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        ssd[blockX] = shared_ssd[0];
    }

    return;
}

__global__ void jacobi_iteration_kernel_optimized_coalesced(float *A, float *B, float *x, float *new_x, double *ssd, int num_rows, int num_columns)
{
    extern __shared__ double shared_ssd[];

    int blockX = blockIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (col < num_columns) {
        double old_value = x[col];

        double sum = 0.0;
        for (int row = 0; row < num_rows; row++) {
            if (row != col) {
                int idx = col + row*num_columns;
                sum += A[idx] * x[row];
            }
        }

        double new_value = (B[col] - sum) / A[col * num_rows + col];

        new_x[col] = new_value;

        double diff = old_value - new_value;
        double local_ssd = diff * diff;

        shared_ssd[tid] = local_ssd;
    }
    else {
        shared_ssd[tid] = 0.0;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            shared_ssd[tid] += shared_ssd[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        ssd[blockX] = shared_ssd[0];
    }

    return;
}
