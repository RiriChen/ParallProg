/* Solve Jacobi using AVX instructions */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "jacobi_solver.h"


/* FIXME: Complete this function to perform the Jacobi calculation using AVX.
 * Result must be placed in avx_solution_x. */
void compute_using_avx(const matrix_t A, matrix_t avx_solution_x, const matrix_t B, int max_iter)
{
    int i, j;
    int num_rows = A.num_rows;
    int num_cols = A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values */
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);

    /* Initialize current Jacobi solution */
    for (i = 0; i < num_rows; i++)
        avx_solution_x.elements[i] = B.elements[i];

    /* Setup the ping-pong buffers */
    float *src = avx_solution_x.elements;
    float *dest = new_x.elements;
    float *temp;

    /* Perform Jacobi iteration. */
    int done = 0;
    double ssd, mse;
    int num_iter = 0;

    while (!done) {
        for (i = 0; i < num_rows; i++) {
            double sum = -A.elements[i * num_cols + i] * src[i];
            __m256 vsum = _mm256_setzero_ps();

            for (j = 0; j < num_cols; j += 8) {
                __m256 va = _mm256_loadu_ps(&A.elements[i * num_cols + j]);
                __m256 vx = _mm256_loadu_ps(&src[j]);
                __m256 vprod = _mm256_mul_ps(va, vx);
                vsum = _mm256_add_ps(vsum, vprod);
            }

            float sum_arr[8];
            _mm256_storeu_ps(sum_arr, vsum);
            sum += sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                   sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

            /* Update values for the unkowns for the current row. */
            dest[i] = (B.elements[i] - sum) / A.elements[i * num_cols + i];
        }

        /* Check for convergence and update the unknowns. */
        ssd = 0.0;
        for (i = 0; i < num_rows; i++) {
            ssd += (dest[i] - src[i]) * (dest[i] - src[i]);
        }

        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        //fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse);

        if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;

        /* Flip the ping-pong buffers */
        temp = src;
        src = dest;
        dest = temp;
    }

    /*
    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");
    */

    free(new_x.elements);
}
