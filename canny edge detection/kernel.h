//Copyright (C) Sorin Draghici <dsorin95@gmail.com>
#ifndef _KERNEL_H
#define _KERNEL_H

#ifdef __cplusplus
extern "C"
#endif

void cannyGPU(float *im, float *image_out,
float *NR, float *G, float *phi, float *Gx, float *Gy, int *pedge,
float level,
int height, int width);

#endif
