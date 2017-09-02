//Copyright (C) Sorin Draghici <dsorin95@gmail.com>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "kernel.h"

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

void cannyCPU(float *im, float *image_out, float *NR, float *G, float *phi, float *Gx, float *Gy, int *pedge, float level, int height, int width);

double get_time()
{
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0, (struct timezone*)0);
	time_ = (double)((tv0.tv_usec + (tv0.tv_sec) * 1000000));
	time = time_ / 1000000;

	return time;
}

//Read a BMP file into memory
unsigned char *readBMP(char *file_name, char header[54], int *w, int *h)
{
	//Open image file in read mode
	FILE *f = fopen(file_name, "rb");
	
	if (!f)
	{
		perror(file_name);
		exit(1);
	}
	
	//Image header
	//Get the amount of read bytes
	int n = fread(header, 1, 54, f);
	
	//If 54 bytes haven't been read, the input image is too small
	if (n != 54)
	{
		fprintf(stderr, "Very small input image (%d bytes)\n", n);
		exit(1);
	}
	
	//First 2 bytes must be a B and M, otherwise it's not a BMP file.
	if (header[0] != 'B' || header[1] != 'M')
	{
		fprintf(stderr, "Input image is not a  BMP file\n");
		exit(1);
	}
	
	//The real image size is the value of the position 2 in the header, minus the 54 bytes from the header
	int imagesize = *(int*)(header + 2) - 54;

	//Valid image only if it has a positive value size or it's below to 48MB
	if (imagesize <= 0 || imagesize > 0x3000000)
	{
		fprintf(stderr, "Input image is too big %d bytes\n", imagesize);
		exit(1);
	}
	
	//If the header is not of size 54 bytes, or the amount of bits per pixel is not 24, the image cannot be used
	if (*(int*)(header + 10) != 54 || *(short*)(header + 28) != 24)
	{
		fprintf(stderr, "Input image is not 24-bit color\n");
		exit(1);
	}
	
	//If the position 30 in the header is not 0, the file is compressed
	if (*(int*)(header + 30) != 0)
	{
		fprintf(stderr, "File compression not supported\n");
		exit(1);
	}
	
	//Get the width and height of the image
	int width = *(int*)(header + 18);
	int height = *(int*)(header + 22);


	//Read the image
	unsigned char *image = (unsigned char*)malloc(imagesize + 256 + width * 6);
	unsigned char *tmp;
	
	image += 128 + width * 3;
	
	if ((n = fread(image, 1, imagesize + 1, f)) != imagesize)
	{
		fprintf(stderr, "File size incorrect: %d bytes read insted of %d\n", n, imagesize);
		exit(1);
	}
	
	fclose(f);
	printf("Image read correctly (width=%i height=%i, imagesize=%i).\n", width, height, imagesize);

	/* Output variables */
	*w = width;
	*h = height;

	return image;
}

//Write a BMP file from memory to a file
void writeBMP(float *imageFLOAT, char *file_name, char header[54], int width, int height)
{
	FILE *f;
	int i, n;

	int imagesize = *(int*)(header + 2) - 54;
	unsigned char *image = (unsigned char*)malloc(3 * sizeof(unsigned char) * width * height);

	for (i = width * height - 1; i >= 0; i--)
	{
		image[3 * i + 2] = imageFLOAT[i]; //B
		image[3 * i + 1] = imageFLOAT[i]; //G
		image[3 * i] = imageFLOAT[i]; //R 
	}

	f = fopen(file_name, "wb");
	
	if (!f)
	{
		perror(file_name);
		exit(1);
	}

	n = fwrite(header, 1, 54, f); //Write the image header
	n += fwrite(image, 1, imagesize, f); //Write the rest of the image
	
	if (n != 54 + imagesize) //The amount of bytes written must be the same as the original (imagesize + 54)
		fprintf(stderr, "Written %d of %d bytes\n", n, imagesize + 54);
		
	fclose(f);
	free(image);
}

//Convert image to black and white
float *RGB2BW(unsigned char *imageUCHAR, int width, int height)
{
	int i, j;
	float *imageBW = (float *)malloc(sizeof(float)*width*height);

	unsigned char R, B, G;

	for (i = 0; i<height; i++)
		for (j = 0; j<width; j++)
		{
			R = imageUCHAR[3 * (i * width + j)];
			G = imageUCHAR[3 * (i * width + j) + 1];
			B = imageUCHAR[3 * (i * width + j) + 2];

			imageBW[i * width + j] = 0.2989 * R + 0.5870 * G + 0.1140 * B;
		}

	return imageBW;
}

//Apply Canny filter to an image. Calculations on CPU
void cannyCPU(float *im, float *image_out, float *NR, float *G, float *phi, float *Gx, float *Gy, int *pedge, float level, int height, int width)
{
	int i, j;
	int ii, jj;
	float PI = 3.141593;

	float lowthres, hithres;

	//Noise reduction
	for (i = 2; i < height - 2; i++)
		for (j = 2; j < width - 2; j++)
		{
			NR[i*width + j] =
				(2.0 * im[(i - 2) * width + (j - 2)] + 4.0 * im[(i - 2) * width + (j - 1)] + 5.0 * im[(i - 2) * width + (j)] + 4.0 * im[(i - 2) * width + (j + 1)] + 2.0 * im[(i - 2) * width + (j + 2)]
			   + 4.0 * im[(i - 1) * width + (j - 2)] + 9.0 * im[(i - 1) * width + (j - 1)] + 12.0* im[(i - 1) * width + (j)] + 9.0 * im[(i - 1) * width + (j + 1)] + 4.0 * im[(i - 1) * width + (j + 2)]
			   + 5.0 * im[(i)     * width + (j - 2)] + 12.0* im[(i) 	* width + (j - 1)] + 15.0* im[(i)	  * width + (j)] + 12.0* im[(i)		* width + (j + 1)] + 5.0 * im[(i)	  * width + (j + 2)]
			   + 4.0 * im[(i + 1) * width + (j - 2)] + 9.0 * im[(i + 1) * width + (j - 1)] + 12.0* im[(i + 1) * width + (j)] + 9.0 * im[(i + 1) * width + (j + 1)] + 4.0 * im[(i + 1) * width + (j + 2)]
			   + 2.0 * im[(i + 2) * width + (j - 2)] + 4.0 * im[(i + 2) * width + (j - 1)] + 5.0 * im[(i + 2) * width + (j)] + 4.0 * im[(i + 2) * width + (j + 1)] + 2.0 * im[(i + 2) * width + (j + 2)]) / 159.0;
		}

	//Intensity gradient of the image
	for (i = 2; i < height - 2; i++)
		for (j = 2; j < width - 2; j++)
		{
			Gx[i * width + j] =
				(1.0 * NR[(i - 2) * width + (j - 2)] + 2.0 * NR[(i - 2) * width + (j - 1)] + (-2.0) * NR[(i - 2) * width + (j + 1)] + (-1.0) * NR[(i - 2) * width + (j + 2)]
				+ 4.0* NR[(i - 1) * width + (j - 2)] + 8.0 * NR[(i - 1) * width + (j - 1)] + (-8.0) * NR[(i - 1) * width + (j + 1)] + (-4.0) * NR[(i - 1) * width + (j + 2)]
				+ 6.0* NR[(i)	  * width + (j - 2)] + 12.0* NR[(i)	    * width + (j - 1)] + (-12.0)* NR[(i)	 * width + (j + 1)] + (-6.0) * NR[(i)	  * width + (j + 2)]
				+ 4.0* NR[(i + 1) * width + (j - 2)] + 8.0 * NR[(i + 1) * width + (j - 1)] + (-8.0) * NR[(i + 1) * width + (j + 1)] + (-4.0) * NR[(i + 1) * width + (j + 2)]
				+ 1.0* NR[(i + 2) * width + (j - 2)] + 2.0 * NR[(i + 2) * width + (j - 1)] + (-2.0) * NR[(i + 2) * width + (j + 1)] + (-1.0) * NR[(i + 2) * width + (j + 2)]);


			Gy[i * width + j] =
				((-1.0) * NR[(i - 2) * width + (j - 2)] + (-4.0) * NR[(i - 2) * width + (j - 1)] + (-6.0) * NR[(i - 2) * width + (j)] + (-4.0) * NR[(i - 2) * width + (j + 1)] + (-1.0) * NR[(i - 2) * width + (j + 2)]
				+ (-2.0)* NR[(i - 1) * width + (j - 2)] + (-8.0) * NR[(i - 1) * width + (j - 1)] + (-12.0)* NR[(i - 1) * width + (j)] + (-8.0) * NR[(i - 1) * width + (j + 1)] + (-2.0) * NR[(i - 1) * width + (j + 2)]
				+ 2.0   * NR[(i + 1) * width + (j - 2)] + 8.0 	 * NR[(i + 1) * width + (j - 1)] + 12.0   * NR[(i + 1) * width + (j)] + 8.0	   * NR[(i + 1) * width + (j + 1)] + 2.0 	* NR[(i + 1) * width + (j + 2)]
				+ 1.0	* NR[(i + 2) * width + (j - 2)] + 4.0	 * NR[(i + 2) * width + (j - 1)] + 6.0	  * NR[(i + 2) * width + (j)] + 4.0    * NR[(i + 2) * width + (j + 1)] + 1.0    * NR[(i + 2) * width + (j + 2)]);

			G[i * width + j] = sqrtf((Gx[i * width + j] * Gx[i * width + j]) + (Gy[i * width + j] * Gy[i * width + j]));
			phi[i * width + j] = atan2f(fabs(Gy[i * width + j]), fabs(Gx[i * width + j]));

			if(fabs(phi[i * width + j]) <= PI / 8)
				phi[i * width + j] = 0;
			else
				if(fabs(phi[i * width + j]) <= 3 * (PI / 8))
					phi[i*width + j] = 45;
				else
					if(fabs(phi[i * width + j]) <= 5 * (PI / 8))
						phi[i * width + j] = 90;
					else
						if (fabs(phi[i * width + j]) <= 7 * (PI / 8))
							phi[i * width + j] = 135;
						else phi[i*width + j] = 0;
		}

	//Edge
	for(i = 3; i < height - 3; i++)
		for(j = 3; j < width - 3; j++)
		{
			pedge[i * width + j] = 0;
			
			if(phi[i * width + j] == 0)
			{
				if(G[i * width + j] > G[i * width + j + 1] && G[i * width + j] > G[i * width + j - 1]) //edge is in N-S
					pedge[i * width + j] = 1;
			}
			else
				if(phi[i * width + j] == 45)
				{
					if(G[i * width + j] > G[(i + 1) * width + j + 1] && G[i * width + j] > G[(i - 1) * width + j - 1]) // edge is in NW-SE
						pedge[i * width + j] = 1;
				}
				else
					if(phi[i * width + j] == 90)
					{
						if(G[i * width + j] > G[(i + 1) * width + j] && G[i * width + j] > G[(i - 1) * width + j]) //edge is in E-W
							pedge[i * width + j] = 1;
					}
					else
						if(phi[i * width + j] == 135)
						{
							if (G[i * width + j] > G[(i + 1) * width + j - 1] && G[i * width + j] > G[(i - 1) * width + j + 1]) // edge is in NE-SW
								pedge[i * width + j] = 1;
						}
		}

	// Hysteresis Thresholding
	lowthres = level / 2;
	hithres = 2 * level;

	for(i = 0; i < height; i++)
		for(j = 0; j < width; j++)
			image_out[i * width + j] = 0;

	for(i = 3; i < height - 3; i++)
		for(j = 3; j < width - 3; j++)
		{
			if(G[i * width + j] > hithres && pedge[i * width + j])
				image_out[i * width + j] = 255;
			else
				if(pedge[i * width + j] && G[i * width + j] >= lowthres && G[i * width + j] < hithres)
				{
					//check neighbours 3x3
					for(ii = -1; ii <= 1; ii++)
						for(jj = -1; jj <= 1; jj++)
							if(G[(i + ii) * width + j + jj] > hithres)
								image_out[i * width + j] = 255;
				}
		}
}

int main(int argc, char **argv)
{
	int width, height;
	unsigned char *imageUCHAR;
	float *imageBW;
	char header[54];

	//Time related variables
	double t0, t1;
	double cpu_time_used = 0.0;

	//Must have 3 argument values
	if(argc < 4)
	{
		fprintf(stderr, "Use parameters: exe  input.bmp output.bmp [cg]\n");
		exit(1);
	}

	//Read image & Convert image
	imageUCHAR = readBMP(argv[1], header, &width, &height);
	imageBW = RGB2BW(imageUCHAR, width, height);

	// Aux. memory
	float *NR = (float *)malloc(sizeof(float) * width * height);
	float *G  = (float *)malloc(sizeof(float) * width * height);
	float *phi= (float *)malloc(sizeof(float) * width * height);
	float *Gx = (float *)malloc(sizeof(float) * width * height);
	float *Gy = (float *)malloc(sizeof(float) * width * height);
	int   *pedge = (int *)malloc(sizeof(int) * width * height);
	float *imageOUT = (float *)malloc(sizeof(float) * width * height);

	clock_t ini, fin;

	switch (argv[3][0])
	{
	case 'c':
		t0 = get_time();
		
		cannyCPU(imageBW, imageOUT, NR, G, phi, Gx, Gy, pedge, 1000.0, height, width);
		
		t1 = get_time();
		
		printf("CPU Exection time %f ms.\n", t1 - t0);
		break;
	case 'g':
		cannyGPU(imageBW, imageOUT, NR, G, phi, Gx, Gy, pedge, 1000.0, height, width);
		break;
	default:
		printf("Not Implemented yet!!\n");
	}

	//Write processed image to file
	writeBMP(imageOUT, argv[2], header, width, height);

	//Free allocated memory
	free(imageBW);
	free(imageOUT);
	free(NR);
	free(G);
	free(phi);
	free(Gx);
	free(Gy);
	free(pedge);
}
