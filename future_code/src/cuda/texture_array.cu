#include "texture_array.h"

TextureArray::~TextureArray()
{
  if (num_images > 0) {
    for (int i = 0; i < num_images; ++i) {
      checkCudaErrors(cudaDestroyTextureObject(data[i]));
      checkCudaErrors(cudaFreeArray(arrays[i]));
    }
  }
}

void TextureArray::setDataToTextureMemory(const std::vector<Mat2D>& mat)
{
  num_images = mat.size();

  for (int i = 0; i < num_images; i++) {
    const int rows = mat[i].height;
    const int cols = mat[i].width;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&arrays[i], &channelDesc, cols, rows));
    checkCudaErrors(cudaMemcpy2DToArray(
        arrays[i], 0, 0, mat[i].data.ptr(), mat[i].data.step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arrays[i];

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    checkCudaErrors(cudaCreateTextureObject(&(data[i]), &resDesc, &texDesc, nullptr));
  }
}