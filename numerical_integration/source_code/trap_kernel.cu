/* GPU kernel to estimate integral of the provided function using the trapezoidal rule. */

/* Device function which implements the function. Device functions can be called from within other __device__ functions or __global__ functions (the kernel), but cannot be called from the host. */

#define THREAD_BLOCK_SIZE 1024

__device__ float fd(float x)
{
     return sqrtf((1 + x*x)/(1 + x*x*x*x));
}

/* Kernel function */
__global__ void trap_kernel(float a, float b, float h, int n, double *result)
{
    __shared__ double partial_sum[THREAD_BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    double sum = 0.0;
    unsigned int i = tid;

    if (i > 0) {
        while (i < n) {
            sum += fd(a + i * h);
            i += stride;
        }
    }

    partial_sum[threadIdx.x] = sum;
    __syncthreads();

    i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
        }
        __syncthreads();

        i /= 2;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(result, (fd(a) + fd(b)) / 2.0);
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, partial_sum[0]);
    }
}

