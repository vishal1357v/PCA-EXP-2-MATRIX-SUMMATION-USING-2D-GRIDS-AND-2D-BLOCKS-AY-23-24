# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks

<h3>ENTER YOUR NAME  : VISHAL P</h3>
<h3>ENTER YOUR REGISTER NO : 212224230306</h3>
<h3>EX. NO : 02</h3>
<h3>DATE : 04-08-2026</h3>
<h2> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h2>
    
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu

ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:

To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:

Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler


## PROCEDURE:

- Set up CUDA device – Initialize the GPU and load device properties.

- Create host matrices – Allocate integer arrays for A, B, and C.

- Fill input data – Generate random values for matrices A and B.

- Do CPU computation – Add A and B on the host to form a reference result.

- Allocate device buffers – Reserve GPU memory for A, B, and C.

- Transfer inputs to GPU – Copy A and B from host to device memory.

- Define execution config – Set block size to 16×16 and compute the grid size.

- Execute kernel – Launch GPU threads to perform element-wise matrix addition.

- Copy results back – Retrieve output matrix C from device to host.

- Validate and release memory – Compare results and free host/device resources.

## PROGRAM:
```py
%%writefile sumMatrix2D.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

// ---------------- INITIALIZE INTEGER DATA ----------------
void initialData(int *ip, const int size)
{
    for (int i = 0; i < size; i++)
        ip[i] = rand() % 100;  // integer values only
}

// ---------------- HOST MATRIX SUM ----------------
void sumMatrixOnHost(int *A, int *B, int *C, int nx, int ny)
{
    for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++)
        {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
}

// ---------------- COMPARE GPU + CPU RESULTS ----------------
void checkResult(int *hostRef, int *gpuRef, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            printf("Mismatch at %d: host %d gpu %d\n", 
                    i, hostRef[i], gpuRef[i]);
            return;
        }
    }
    printf("Arrays match!\n");
}

// ---------------- GPU KERNEL ----------------
__global__ void sumMatrixOnGPU2D(int *A, int *B, int *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        int idx = iy * NX + ix;
        C[idx] = A[idx] + B[idx];
    }
}

// ---------------- MAIN FUNCTION ----------------
int main()
{
    int nx = 1 << 12;  // 4096
    int ny = 1 << 12;  // 4096
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    printf("Matrix Summation using 2D Grid + 2D Block\n");
    printf("Matrix size: %d x %d = %d elements\n", nx, ny, nxy);

    // Allocate host memory
    int *h_A = (int*)malloc(nBytes);
    int *h_B = (int*)malloc(nBytes);
    int *hostRef = (int*)malloc(nBytes);
    int *gpuRef  = (int*)malloc(nBytes);

    // Initialize matrices
    double start = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    printf("Data initialization: %f sec\n", seconds() - start);

    // CPU sum
    start = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    printf("CPU execution time: %f sec\n", seconds() - start);

    // GPU memory allocation
    int *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, nBytes));
    CHECK(cudaMalloc((void**)&d_B, nBytes));
    CHECK(cudaMalloc((void**)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // ------- FIXED BLOCK SIZE -------
    dim3 block(16, 16);  // SAFE SIZE (256 threads/block)
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    printf("Launching kernel <<<(%d,%d), (%d,%d)>>>\n",
           grid.x, grid.y, block.x, block.y);

    // GPU kernel execution
    start = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);

    // Check kernel error
    CHECK(cudaGetLastError());       // <---- IMPORTANT FIX
    CHECK(cudaDeviceSynchronize());
    printf("GPU execution time: %f sec\n", seconds() - start);

    // Copy back result
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Compare CPU + GPU results
    checkResult(hostRef, gpuRef, nxy);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return 0;
}

!nvcc -arch=sm_70 sumMatrix2D.cu -o sumMatrix2D
!./sumMatrix2D

```


## OUTPUT:

<img width="430" height="159" alt="image" src="https://github.com/user-attachments/assets/9a993364-9911-4222-9559-431a5926020b" />


## RESULT:
Matrix addition using a 2D grid and 2D blocks was successfully implemented on the GPU.
The GPU-computed matrix exactly matched the CPU result, demonstrating correct execution and parallel speedup.
