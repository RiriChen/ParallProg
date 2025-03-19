/* Host code for the Jacobi method of solving a system of linear equations
 * by iteration.

 * Build as follows: make clean && make

 * Author: Naga Kandasamy
 * Date modified: March 6, 2025
 *
 * Student name(s); Ricky Chen, Keith Trungcao
 * Date modified: 3/13/25
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

struct timeval start, stop;

/* Uncomment the line below if you want the code to spit out debug information. */
/* #define DEBUG */

int main(int argc, char **argv)
{
    if (argc > 1) {
        printf("This program accepts no arguments\n");
        exit(EXIT_FAILURE);
    }

    matrix_t  A;                    /* N x N constant matrix */
    matrix_t  B;                    /* N x 1 b matrix */
    matrix_t reference_x;           /* Reference solution */
    matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

    /* Initialize the random number generator */
    srand(time(NULL));

    /* Generate diagonally dominant matrix */
    printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
    A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
    if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
    }

    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
    reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
    print_matrix(A);
    print_matrix(B);
    print_matrix(reference_x);
#endif

    /* Compute Jacobi solution on CPU */
    printf("\nPerforming Jacobi iteration on the CPU\n");
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */

    /* Compute Jacobi solution on device. Solutions are returned
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
    compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B);
    printf("Naive jacobi:\n");

    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */

    printf("optimzied jacobi:\n");

    display_jacobi_solution(A, gpu_opt_solution_x, B);

    free(A.elements);
    free(B.elements);
    free(reference_x.elements);
    free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform Jacobi calculation on device */
void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x,
                       matrix_t gpu_opt_sol_x, const matrix_t B)
{
    matrix_t A_d = allocate_matrix_on_device(A);
    matrix_t B_d = allocate_matrix_on_device(B);

    printf("\nRunning Naive Implementation\n");
    gettimeofday(&start, NULL);
    matrix_t x_naive_d = allocate_matrix_on_device(gpu_naive_sol_x);
    matrix_t new_x_naive_d = allocate_matrix_on_device(gpu_naive_sol_x);

    double *ssd_naive_d;
    cudaMalloc((void **)&ssd_naive_d, sizeof(double) * ((A.num_rows + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE));

    copy_matrix_to_device(A_d, A);
    copy_matrix_to_device(B_d, B);
    copy_matrix_to_device(x_naive_d, gpu_naive_sol_x);

    dim3 threads(THREAD_BLOCK_SIZE, 1, 1);
    dim3 grid((A.num_rows + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);

    int done_naive = 0;
    double ssd_naive = 0.0;

    while (!done_naive) {
        jacobi_iteration_kernel_naive<<<grid, threads, THREAD_BLOCK_SIZE * sizeof(double)>>>(
            A_d.elements, B_d.elements, x_naive_d.elements, new_x_naive_d.elements, ssd_naive_d, A.num_rows, A.num_columns);

        double *ssd_naive_h = (double *)malloc(grid.x * sizeof(double));
        cudaMemcpy(ssd_naive_h, ssd_naive_d, grid.x * sizeof(double), cudaMemcpyDeviceToHost);

        ssd_naive = 0.0;
        for (int i = 0; i < grid.x; i++) {
            ssd_naive += ssd_naive_h[i];
        }
        free(ssd_naive_h);

        //printf("ssd_naive: %lf\n", sqrt(ssd_naive));
        //threshold 1-e5
        if (sqrt(ssd_naive) <= THRESHOLD) {
            //printf("done naiive\n");
            done_naive = 1;
        }

        matrix_t tmp = x_naive_d;
        x_naive_d = new_x_naive_d;
        new_x_naive_d = tmp;
    }

    gettimeofday(&stop, NULL);
    fprintf(stderr, "Naive execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    check_CUDA_error("Error in kernel");

    copy_matrix_from_device(gpu_naive_sol_x, x_naive_d);


    //matrix_t A_d = allocate_matrix_on_device(A);
    //fprintf(stderr, "\nConverting A to B = A^T to get elements ordered in column-major fashion\n");

    printf("\nRunning coalesced Implementation\n");
    // printf("This is A_T\n");
    // print_matrix(A_T);
    gettimeofday(&start, NULL);

    matrix_t A_T  = allocate_matrix_on_host(A.num_rows, A.num_columns, 0);
    int i, j;
    for (i = 0; i < A_T.num_rows; i++) {
       for (j = 0; j < A_T.num_columns; j++) {
        A_T.elements[i * A_T.num_columns + j] = A.elements[j * A_T.num_columns + i];
       }
    }

    matrix_t A_T_gpu = allocate_matrix_on_device(A_T);

    matrix_t x_opt_d = allocate_matrix_on_device(gpu_opt_sol_x);
    matrix_t new_x_opt_d = allocate_matrix_on_device(gpu_opt_sol_x);

    double *ssd_opt_d;
    cudaMalloc((void **)&ssd_opt_d, sizeof(double) * ((A.num_rows + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE));

    copy_matrix_to_device(A_T_gpu, A_T);
    copy_matrix_to_device(B_d, B);
    copy_matrix_to_device(x_opt_d, gpu_opt_sol_x);


    int done_opt = 0;
    double ssd_opt = 0.0;


    while (!done_opt) {

        jacobi_iteration_kernel_optimized_coalesced<<<grid, threads, THREAD_BLOCK_SIZE * sizeof(double)>>>(
            A_T_gpu.elements, B_d.elements, x_opt_d.elements, new_x_opt_d.elements, ssd_opt_d, A_T_gpu.num_rows, A_T_gpu.num_columns);

        double *ssd_opt_h = (double *)malloc(grid.x * sizeof(double));
        cudaMemcpy(ssd_opt_h, ssd_opt_d, grid.x * sizeof(double), cudaMemcpyDeviceToHost);

        ssd_opt = 0.0;
        for (int i = 0; i < grid.x; i++) {
            ssd_opt += ssd_opt_h[i];
        }
        free(ssd_opt_h);

        if (sqrt(ssd_opt) <= THRESHOLD) {
            done_opt = 1;
        }

        matrix_t tmp = x_opt_d;
        x_opt_d = new_x_opt_d;
        new_x_opt_d = tmp;
    }

    gettimeofday(&stop, NULL);
    fprintf(stderr, "Coalesced execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    check_CUDA_error("Error in kernel");

    // /* Copy result from the device */
    copy_matrix_from_device(gpu_opt_sol_x, x_opt_d);

    /* Free memory on host */
    free((void *)A_T.elements);
    cudaFree(A_T_gpu.elements);
    cudaFree(B_d.elements);
    cudaFree(x_opt_d.elements);
    cudaFree(new_x_opt_d.elements);
    cudaFree(ssd_opt_d);
    cudaFree(A_d.elements);
    //cudaFree(B_d.elements);
    cudaFree(x_naive_d.elements);
    cudaFree(new_x_naive_d.elements);
    cudaFree(ssd_naive_d);
    return;
}


/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

    M.elements = (float *)malloc(size * sizeof(float));
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }

    return M;
}

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
    for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
            printf("%f ", M.elements[i * M.num_columns + j]);
        }

        printf("\n");
    }

    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
    return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err) {
        printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    unsigned int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

    /* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
    for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);

    /* Make diagonal entries large with respect to the entries on each row. */
    for (i = 0; i < num_rows; i++) {
        float row_sum = 0.0;
        for (j = 0; j < num_columns; j++) {
            row_sum += fabs(M.elements[i * M.num_rows + j]);
        }

        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
    }

    return M;
}
