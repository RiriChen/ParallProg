/* Optimize Jacobi using pthread and AVX instructions */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>
#include "jacobi_solver.h"

#define VECTOR_SIZE 8 /* AVX operates on 8 single-precision floating-point values */

typedef struct {
    int tid;                   /* Thread ID */
    int num_threads;           /* Total number of threads */
    int num_rows;              /* Number of rows in the matrix */
    int num_cols;              /* Number of columns */
    float *A;                  /* Pointer to matrix A */
    float *B;                  /* Pointer to matrix B */
    float *x;                  /* Pointer to matrix x */
    float *new_x;              /* Pointer to matrix new_x */
    int num_iter;              /* Shared number of iterations */
    int max_iter;              /* Max iterations */
    int chunk_size;            /* Chunk size */
} thread_data_t;

/* Worker function for Pthreads + AVX Jacobi computation */
void *jacobi_pthread_avx_worker(void *arg)
{
    thread_data_t *thread_data = (thread_data_t *)arg;
    int num_cols = thread_data->num_cols;
    int num_rows = thread_data->num_rows;
    int start = thread_data->tid * thread_data->chunk_size;
    int end = start + thread_data->chunk_size;

    int i;

    float *src = thread_data->x;
    float *dest = thread_data->new_x;
    float *temp;

    int done = 0;
    double ssd, mse;

    while (!done) {
        for (int i = start; i < end; i++) {
            double sum = -thread_data->A[i * num_cols + i] * src[i];
            __m256 vsum = _mm256_setzero_ps();

            for (int j = 0; j <= num_cols - VECTOR_SIZE; j += VECTOR_SIZE) {
                __m256 va = _mm256_loadu_ps(&thread_data->A[i * num_cols + j]);
                __m256 vx = _mm256_loadu_ps(&src[j]);
                __m256 vprod = _mm256_mul_ps(va, vx);
                vsum = _mm256_add_ps(vsum, vprod);
            }

            float sum_arr[VECTOR_SIZE];
            _mm256_storeu_ps(sum_arr, vsum);
            sum += sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                   sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

            dest[i] = (thread_data->B[i] - sum) / thread_data->A[i * num_cols + i];
        }

        /* Check for convergence and update the unknowns. */
        ssd = 0.0;
        __m256 vssd = _mm256_setzero_ps();
        for (i = 0; i < num_rows; i += 8) {
            __m256 vdest = _mm256_loadu_ps(&dest[i]);
            __m256 vsrc = _mm256_loadu_ps(&src[i]);
            __m256 dest_src_differenece = _mm256_sub_ps(vdest, vsrc);
            __m256 dest_src_differenece_squared = _mm256_mul_ps(dest_src_differenece, dest_src_differenece);
            vssd =  _mm256_add_ps(vssd, dest_src_differenece_squared);
        }

        float ssd_arr[8];
        _mm256_storeu_ps(ssd_arr, vssd);
        ssd += ssd_arr[0] + ssd_arr[1] + ssd_arr[2] + ssd_arr[3] +
               ssd_arr[4] + ssd_arr[5] + ssd_arr[6] + ssd_arr[7];

        thread_data->num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        //fprintf(stderr, "Iteration: %d. MSE = %f\n", thread_data->num_iter, mse);

        if (mse <= THRESHOLD || thread_data->num_iter == thread_data->max_iter) {
            done = 1;
        }

        /* Flip the ping-pong buffers */
        temp = src;
        src = dest;
        dest = temp;
    }

    pthread_exit(NULL);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads + AVX.
 * Result must be placed in pthread_avx_solution_x. */
void compute_using_pthread_avx(const matrix_t A, matrix_t pthread_avx_solution_x, const matrix_t B, int max_iter, int num_threads)
{
    int i;
    int num_rows = A.num_rows;
    int num_cols = A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values */
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);

    /* Initialize current Jacobi solution */
    for (i = 0; i < num_rows; i++)
        pthread_avx_solution_x.elements[i] = B.elements[i];

    /* Thread setup */
    pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    int chunk_size = (int)floor((float)num_rows/(float) num_threads);

    thread_data_t *thread_data = (thread_data_t *)malloc(num_threads * sizeof(thread_data_t));
    int num_iter = 0;

    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_rows = num_rows;
        thread_data[i].num_cols = num_cols;
        thread_data[i].A = A.elements;
        thread_data[i].B = B.elements;
        thread_data[i].x = pthread_avx_solution_x.elements;
        thread_data[i].new_x = new_x.elements;
        thread_data[i].num_iter = num_iter;
        thread_data[i].max_iter = max_iter;
        thread_data[i].chunk_size = chunk_size;
    }

    for (i = 0; i < num_threads; i++) {
        pthread_create(&thread_id[i], NULL, jacobi_pthread_avx_worker, (void *)&thread_data[i]);
    }

    for (i = 0; i < num_threads; i++) {
        pthread_join(thread_id[i], NULL);
    }

    /*
    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");
    */

    free((void *)thread_data);
    free((void *)thread_id);
    free(new_x.elements);
}
