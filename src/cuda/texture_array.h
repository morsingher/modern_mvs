#ifndef MVS_TEXTURE_ARRAY_H
#define MVS_TEXTURE_ARRAY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include "cuda_helper.h"
#include "mat.h"

// TODO: Can this be removed? How?

#define MAX_IMAGES 16

class TextureArray {
 public:
  TextureArray() : num_images(0){};
  ~TextureArray();

  void setDataToTextureMemory(const std::vector<Mat2D>& mat);

  cudaTextureObject_t data[MAX_IMAGES];
  cudaArray* arrays[MAX_IMAGES];
  int num_images;
};

#endif