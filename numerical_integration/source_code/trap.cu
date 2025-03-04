/* Reference code implementing numerical integration.
 *
 * Build and execute as follows:
        make clean && make
        ./trap a b n

 * Author: Naga Kandasamy
 * Date modified: February 28, 2025

 * Student name(s): Ricky Chen, Keith Trungcao
 * Date modified: 3/2/25
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

/* Include the kernel code */
#include "trap_kernel.cu"

double compute_on_device(float, float, int, float);
extern "C" double compute_gold(float, float, int, float);

struct timeval start, stop;

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s a b n\n", argv[0]);
        fprintf(stderr, "a: Start limit. \n");
        fprintf(stderr, "b: end limit\n");
        fprintf(stderr, "n: number of trapezoids\n");
        exit(EXIT_FAILURE);
    }

    int a = atoi(argv[1]); /* Left limit */
    int b = atoi(argv[2]); /* Right limit */
    int n = atoi(argv[3]); /* Number of trapezoids */

    float h = (b-a)/(float)n; // Height of each trapezoid
    printf("Number of trapezoids = %d\n", n);
    printf("Height of each trapezoid = %f \n", h);

    gettimeofday(&start, NULL);
    double reference = compute_gold(a, b, n, h);
    gettimeofday(&stop, NULL);
    printf("Reference solution computed on the CPU = %f \n", reference);
    printf("CPU execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Write this function to complete the trapezoidal on the GPU. */
    double gpu_result = compute_on_device(a, b, n, h);
    printf("Solution computed on the GPU = %f \n", gpu_result);
    printf("Kernel execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

}

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_on_device(float a, float b, int n, float h)
{
    double *d_result;
    double h_result = 0.0;

    cudaMalloc((void**)&d_result, sizeof(double));

    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    gettimeofday(&start, NULL);
    trap_kernel<<<num_blocks, threads_per_block>>>(a, b, h, n, d_result);
    cudaDeviceSynchronize();
    gettimeofday(&stop, NULL);

    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    return h_result * h;
}



