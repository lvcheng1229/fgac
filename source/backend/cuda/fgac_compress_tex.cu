#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "fgac_cuda.h"
#include "fgac_compress_texture.cuh"

#define CUDA_VARIFY(expr)\
if (expr != cudaSuccess)\
{\
__debugbreak(); \
}\

__gloabl__ void testKernel(uchar4* outputData, int width, int height, cudaTextureObject_t tex)
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = (float)x - (float)width / 2;
	float v = (float)y - (float)height / 2;

	u /= (float)width;
	v /= (float)height;


	// read from texture and write to global memory
	outputData[y * width + x] = tex2D<uchar4>(tex, u + 0.5f, v + 0.5f);
}

void CudaTestFunc()
{
	CUDA_VARIFY(cudaSetDevice(0));

	std::string imagePath("G:/fgac/build/test.jpeg");
	int width = 0, height = 0, comp = 0;
	stbi_uc* srcData = stbi_load(imagePath.c_str(), &width, &height, &comp, STBI_rgb_alpha);

	uint32_t texSize = width * height * 4 * sizeof(uint8_t);

	// allocate dest image data on deivce memory
	float* destData = nullptr;
	CUDA_VARIFY(cudaMalloc((void**)&destData, texSize));

	// create texture format
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

	// allocate input texture data
	cudaArray* cuArray;
	CUDA_VARIFY(cudaMallocArray(&cuArray, &channelDesc, width, height));
	CUDA_VARIFY(cudaMemcpyToArray(cuArray, 0, 0, srcData, texSize, cudaMemcpyHostToDevice));


	cudaResourceDesc texResDesc;
	memset(&texResDesc, 0, sizeof(cudaResourceDesc));

	texResDesc.resType = cudaResourceTypeArray;
	texResDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));

	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t texObject;
	CUDA_VARIFY(cudaCreateTextureObject(&texObject, &texResDesc, &texDesc, NULL));

	dim3 dimBlock(8, 8, 1);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

	testKernel << <dimGrid, dimBlock, 0 >> > (destData, width, height, texObject);
}
