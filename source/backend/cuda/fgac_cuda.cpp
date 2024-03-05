#include "fgac_cuda.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

#include "stb_image.h"
#include "stb_image_write.h"

#include "fgac_cuda.h"

#define CUDA_VARIFY(expr)\
if (expr != cudaSuccess)\
{\
__debugbreak(); \
}\

extern "C" void testKernel(dim3 gridSize, dim3 blockSize, uchar4 * outputData, int width, int height, cudaTextureObject_t tex);

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

	testKernel(dimGrid, dimBlock, (uchar4*)destData, width, height, texObject);

	CUDA_VARIFY(cudaDeviceSynchronize());
	float* hOutputData = (float*)malloc(texSize);
	CUDA_VARIFY(cudaMemcpy(hOutputData, destData, texSize, cudaMemcpyDeviceToHost));

	std::string outImagePath("G:/fgac/build/otest.tga");
	stbi_write_tga(outImagePath.c_str(), width, height, 4, hOutputData);
}
