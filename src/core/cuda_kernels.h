#ifndef MVS_CUDA_KERNELS_H
#define MVS_CUDA_KERNELS_H

#include "options.h"
#include "plane.h"
#include "texture_array.h"

__global__ void initialize(Plane* planes, const Camera* cameras, float* costs, curandState* rand, const Options* opt,
                           TextureArray* images, unsigned int* views);

__global__ void checkerboardBlack(Plane* planes, const Camera* cameras, float* costs, curandState* rand,
                                  const Options* opt, TextureArray* images, TextureArray* depths, unsigned int* views,
                                  const int iter);

__global__ void checkerboardRed(Plane* planes, const Camera* cameras, float* costs, curandState* rand,
                                const Options* opt, TextureArray* images, TextureArray* depths, unsigned int* views,
                                const int iter);

__global__ void getResults(Plane* planes, const Camera* cameras);

#endif