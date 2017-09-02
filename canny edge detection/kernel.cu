//Copyright (C) Sorin Draghici <dsorin95@gmail.com>
#include <cuda.h>
#include <math.h>
#include "kernel.h"
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

#define DIMBLOCK 32
#define PI 3.141593

double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0, (struct timezone*)0);
	time_ = (double)((tv0.tv_usec + (tv0.tv_sec) * 1000000));
	time = time_ / 1000000;
	return(time);
}

__global__ void calculateNoiseReduction(float *im, float *NR, int width, int height)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y + 2, j = blockIdx.x * blockDim.x + threadIdx.x + 2;

	if(i < height - 2 && j < width - 2)
	{
		NR[i * width + j] =
			(2.0 *im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
			+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
			+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
			+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
			+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)]) / 159.0;
	}
}

__global__ void calculateGradient(float *NR, float *G, float *phi, int width, int height)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y + 2, j = blockIdx.x * blockDim.x + threadIdx.x + 2;

	if(i < height - 2 && j < width - 2)
	{
		float Gx = 
			 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
			+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
			+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
			+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
			+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);

		float Gy = 
			 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
			+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
			+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
			+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

		G[i*width+j]   = sqrtf((Gx*Gx)+(Gy*Gy));
		phi[i*width+j] = atan2f(fabs(Gy),fabs(Gx));

		if(fabs(phi[i*width+j])<=PI/8 )
			phi[i*width+j] = 0;
		else if (fabs(phi[i*width+j])<= 3*(PI/8))
			phi[i*width+j] = 45;
		else if (fabs(phi[i*width+j]) <= 5*(PI/8))
			phi[i*width+j] = 90;
		else if (fabs(phi[i*width+j]) <= 7*(PI/8))
			phi[i*width+j] = 135;
		else phi[i*width+j] = 0;
	}
}

__global__ void calculateEdgesAndHysteresis(float *phi, float *G, float *image_out, float level, int width, int height)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;

	image_out[i * width + j] = 0;

	if(i > 2 && i < height - 3 && j > 2 && j < width - 3)
	{
		float pedge = 0, phiVal = phi[i * width + j];

		if((phiVal == 0 && G[i * width + j] > G[i * width + j + 1] && G[i * width + j] > G[i * width + j - 1])
		|| (phiVal == 45 && G[i * width + j] > G[(i + 1) * width + j + 1] && G[i * width + j] > G[(i - 1) * width + j - 1])
		|| (phiVal == 90 && G[i * width + j] > G[(i + 1) * width + j] && G[i * width + j] > G[(i - 1) * width + j])
		|| (phiVal == 135 && G[i * width + j] > G[(i + 1) * width + j - 1] && G[i * width + j] > G[(i - 1) * width + j + 1]))
				pedge = 1;
	
		float lowthres = level / 2, hithres = level * 2;
		int ii = 0, jj = 0;

		if(G[i * width + j] > hithres && pedge)
			image_out[i * width + j] = 255;
		else
			if(pedge && G[i * width + j] >= lowthres && G[i * width + j] < hithres)
				for(ii = -1; ii <= 1; ii++)
					for(jj = -1; jj <= 1; jj++)
						if(G[(i + ii) * width + j + jj] > hithres)
							image_out[i * width + j] = 255;
	}
}

void cannyGPU(float *im, float *image_out, float *NR, float *G, float *phi, float *Gx, float *Gy, int *pedge, float level, int height, int width)
{
	dim3 dimBlocks(DIMBLOCK, DIMBLOCK);
	dim3 dimGrid(width % DIMBLOCK > 0 ? width / DIMBLOCK + 1 : width / DIMBLOCK, height % DIMBLOCK > 0 ? height / DIMBLOCK + 1 : height / DIMBLOCK);
	float *temp;
	 
	cudaMalloc((void**)&temp, width * height * sizeof(float) * 5);
	cudaMemcpy(&temp[0], im, width * height * sizeof(float), cudaMemcpyHostToDevice);

	double t0, t1;

	t0 = get_time();

	calculateNoiseReduction<<<dimGrid, dimBlocks>>>(&temp[0], &temp[width * height], width, height);
	calculateGradient<<<dimGrid, dimBlocks>>>(&temp[width * height], &temp[width * height * 2], &temp[width * height * 3], width, height);
	calculateEdgesAndHysteresis<<<dimGrid, dimBlocks>>>(&temp[width * height * 3], &temp[width * height * 2], &temp[width * height * 4], level, width, height);
	cudaThreadSynchronize();

	t1 = get_time();

	printf("GPU Exection time %f ms.\n", t1 - t0);

	cudaMemcpy(image_out, &temp[width * height * 4], width * height * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(temp);
}
