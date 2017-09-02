//Copyright (C) Sorin Draghici <dsorin95@gmail.com>
#include <stdio.h>

typedef struct
{
	int width, height, stride;
	float *elements;
}Matrix;

#define BLOCK_SIZE 16

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
	A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;

	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	
	return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// Each thread computes one element of Csub by accumulating results into Cvalue
	float Cvalue = 0.0;

	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;

	// Loop over all the sub-matrices of A and B that are required to compute Csub
	// Multiply each pair of sub-matrices together and accumulate the results
	for(int m = 0; m < (A.width / BLOCK_SIZE); m++)
	{
		// Get sub-matrix Asub of A and Bsub of B
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B, m, blockCol);
		
		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		
		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		
		// Synchronize to make sure the sub-matrices are loaded before starting the computation
		__syncthreads();
		
		// Multiply Asub and Bsub together
		for(int e = 0; e < BLOCK_SIZE; e++)
			Cvalue += As[row][e] * Bs[e][col];
		
		// Synchronize to make sure that the preceding computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	
	// Write Csub to device memory
	// Each thread writes one element
	SetElement(Csub, row, col, Cvalue);
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
	err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n",cudaGetErrorString(err));

	Matrix d_B;
	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
	err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n",cudaGetErrorString(err));

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// Read C from device memory
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	// cudaFree(d_C.elements);
}

void diff(const Matrix A, const Matrix B, const Matrix C)
{
	Matrix C_cpu;

	C_cpu.width = C.width;
	C_cpu.height = C.height;
	C_cpu.elements = (float*)malloc(C_cpu.width * C_cpu.height * sizeof(float));

	int i, j, k;

	//Calculate C in CPU
	for(i = 0; i < A.height; i++)
		for(j = 0; j < B.width; j++)
		{
			C_cpu.elements[i * B.width + j] = 0.0;
			
			for (k = 0; k < A.width; k++)
				C_cpu.elements[i * B.width + j] += A.elements[i * A.width + k] * B.elements[k * B.width + j];
		}

	//Compare C_cpu matrix with C (GPU) matrix 
	for(i = 0; i < C.height; i++)
		for(j = 0; j < C.width; j++)
			if(C_cpu.elements[i * C.width + j] != C.elements[i * C.width + j])
				printf("C_cpu[%i,%i]: %f != %f\n", i, j, C_cpu.elements[i * C.width + j], C.elements[i * C.width + j]);

	free(C_cpu.elements);
}

//Use "name heightA (widthA or heightB) widthB"
int main(int argc, char* argv[])
{
	srand(time(NULL)); //To use rand()

	Matrix A, B, C;
	int a1, a2, b1, b2;

	// Read some values from the commandline
	a1 = atoi(argv[1]); /* Height of A */
	a2 = atoi(argv[2]); /* Width of A */
	b1 = a2; /* Height of B = Width of A */
	b2 = atoi(argv[3]); /* Width of B*/
	
	//Init Matrix A
	A.height = a1;
	A.width = a2;
	A.elements = (float*)malloc(A.width * A.height * sizeof(float));
	
	//Init Matrix B
	B.height = b1;
	B.width = b2;
	B.elements = (float*)malloc(B.width * B.height * sizeof(float));
	
	//Init Matrix C
	C.height = A.height;
	C.width = B.width;
	C.elements = (float*)malloc(C.width * C.height * sizeof(float));
	
	//Randomize Matrix A elements
	for(int i = 0; i < A.height; i++)
		for(int j = 0; j < A.width; j++)
			A.elements[i * A.width + j] = (float)(rand() % 10);

	//Randomize Matrix B elements
	for(int i = 0; i < B.height; i++)
		for(int j = 0; j < B.width; j++)
			B.elements[i * B.width + j] = (float)(rand() % 10);

	//Multiply the matrices
	MatMul(A, B, C);

	// Print up to a 10x10 portion of the three matrices
	printf("Matrix A:\n");
	
	for(int i = 0; i < min(10, A.height); i++)
	{
		for(int j = 0; j < min(10, A.width); j++)
			printf("%f ", A.elements[i*A.width + j]);
	
		printf("\n");
	}

	printf("\n");
	printf("Matrix B:\n");

	for(int i = 0; i < min(10, B.height); i++)
	{
		for(int j = 0; j < min(10, B.width); j++)
			printf("%f ", B.elements[i*B.width + j]);
	
		printf("\n");
	}

	printf("\n");
	printf("Matrix C:\n");

	for(int i = 0; i < min(10, C.height); i++)
	{
		for(int j = 0; j < min(10, C.width); j++)
			printf("%f ", C.elements[i*C.width + j]);
	
		printf("\n");
	}

	printf("\n");

	//Check for differences
	diff(A, B, C);

	free(A.elements);
	free(B.elements);
	free(C.elements);

	printf("Finished\n");
}
