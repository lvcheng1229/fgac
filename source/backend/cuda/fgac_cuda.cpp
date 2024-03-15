#include "fgac_cuda.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <fstream>
#include <assert.h>

#include "stb_image.h"
#include "stb_image_write.h"

#include "fgac_cuda.h"
#include "fgac_compress_texture.h"

#define CUDA_VARIFY(expr)\
if (expr != cudaSuccess)\
{\
__debugbreak(); \
}\

//extern "C" void GPUEncodeKernel(dim3 gridSize, dim3 blockSize, uint8_t * outputData, cudaTextureObject_t tex, fgac_contexti * ctx);
//extern "C" void GPUDecodeKernel(dim3 gridSize, dim3 blockSize, uint8_t* compressedData, uint8_t * decompressedData, fgac_contexti * ctx);

void GPUEncodeKernel(dim3 gridSize, dim3 blockSize, uint8_t * outputData, cudaTextureObject_t tex, fgac_contexti * ctx) {}
void GPUDecodeKernel(dim3 gridSize, dim3 blockSize, uint8_t* compressedData, uint8_t* decompressedData, fgac_contexti* ctx) {}


struct astc_header
{
	uint8_t magic[4];
	uint8_t block_x;
	uint8_t block_y;
	uint8_t block_z;
	uint8_t dim_x[3];			// dims = dim[0] + (dim[1] << 8) + (dim[2] << 16)  texture size
	uint8_t dim_y[3];			// Sizes are given in texels;
	uint8_t dim_z[3];			// block count is inferred
};

static const uint32_t ASTC_MAGIC_ID = 0x5CA1AB13;

static unsigned int unpack_bytes(
	uint8_t a,
	uint8_t b,
	uint8_t c,
	uint8_t d
) {
	return (static_cast<unsigned int>(a)) +
		(static_cast<unsigned int>(b) << 8) +
		(static_cast<unsigned int>(c) << 16) +
		(static_cast<unsigned int>(d) << 24);
}


void EncodeTest()
{
	std::string imagePath("G:/fgac/build/test.jpeg");
	int width = 0, height = 0, comp = 0;
	stbi_uc* srcData = stbi_load(imagePath.c_str(), &width, &height, &comp, STBI_rgb_alpha);
	
	uint32_t texSize = width * height * 4 * sizeof(uint8_t);
	
	// allocate dest image data on deivce memory
	float* destData = nullptr;
	CUDA_VARIFY(cudaMalloc((void**)&destData, texSize));
	
	fgac_contexti* ctx = new fgac_contexti();
	ctx->dim_x = width;
	ctx->dim_y = height;
	ctx->bsd.xdim = 4;
	ctx->bsd.ydim = 4;
	ctx->config.cw_r_weight = 1;
	ctx->config.cw_g_weight = 1;
	ctx->config.cw_b_weight = 1;
	ctx->config.cw_a_weight = 1;
	ctx->config.cw_sum_weight = 4;
	ctx->config.tune_db_limit = 40.5294;
	ctx->config.tune_candidate_limit = 3;
	ctx->config.tune_refinement_limit = 3;
	
	fgac_contexti* pCtx;
	CUDA_VARIFY(cudaMalloc((void**)&pCtx, sizeof(fgac_contexti)));
	CUDA_VARIFY(cudaMemcpy(pCtx, ctx, sizeof(fgac_contexti), cudaMemcpyHostToDevice));
	
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
	printf("aa");
	GPUEncodeKernel(dimGrid, dimBlock, (uint8_t*)destData, texObject, pCtx);
	
	CUDA_VARIFY(cudaDeviceSynchronize());
	float* hOutputData = (float*)malloc(texSize);
	CUDA_VARIFY(cudaMemcpy(hOutputData, destData, texSize, cudaMemcpyDeviceToHost));
	
	std::string outImagePath("G:/fgac/build/otest.tga");
	stbi_write_tga(outImagePath.c_str(), width, height, 4, hOutputData);
}

void DecodeTest()
{
	std::string compressedPath("G:/fgac/build/test.astc");
	std::ifstream file(compressedPath, std::ios::in | std::ios::binary);

	astc_header hdr;
	file.read(reinterpret_cast<char*>(&hdr), sizeof(astc_header));

	unsigned int magicval = unpack_bytes(hdr.magic[0], hdr.magic[1], hdr.magic[2], hdr.magic[3]);
	assert(magicval == ASTC_MAGIC_ID);

	unsigned int block_x = std::max(static_cast<unsigned int>(hdr.block_x), 1u);
	unsigned int block_y = std::max(static_cast<unsigned int>(hdr.block_y), 1u);
	unsigned int block_z = std::max(static_cast<unsigned int>(hdr.block_z), 1u);

	unsigned int dim_x = unpack_bytes(hdr.dim_x[0], hdr.dim_x[1], hdr.dim_x[2], 0);
	unsigned int dim_y = unpack_bytes(hdr.dim_y[0], hdr.dim_y[1], hdr.dim_y[2], 0);
	unsigned int dim_z = unpack_bytes(hdr.dim_z[0], hdr.dim_z[1], hdr.dim_z[2], 0);

	assert(dim_x != 0 || dim_y != 0 || dim_z != 0);

	unsigned int xblocks = (dim_x + block_x - 1) / block_x;
	unsigned int yblocks = (dim_y + block_y - 1) / block_y;
	unsigned int zblocks = (dim_z + block_z - 1) / block_z;
	assert(zblocks == 1);

	size_t data_size = xblocks * yblocks * zblocks * 16;
	uint8_t* buffer = new uint8_t[data_size];

	file.read(reinterpret_cast<char*>(buffer), data_size);

	dim3 dimBlock(8, 8, 1);
	dim3 dimGrid((xblocks + dimBlock.x - 1) / dimBlock.x, (yblocks + dimBlock.y - 1) / dimBlock.y, 1);

	fgac_contexti ctx;
	ctx.dim_x = dim_x;
	ctx.dim_y = dim_y;

	ctx.bsd.xdim = block_x;
	ctx.bsd.ydim = block_y;

	ctx.config.cw_r_weight = 1;
	ctx.config.cw_g_weight = 1;
	ctx.config.cw_b_weight = 1;
	ctx.config.cw_a_weight = 1;
	
	ctx.config.cw_sum_weight = 4;

	ctx.config.tune_db_limit = 40.5294;
	ctx.config.tune_candidate_limit = 3;
	ctx.config.tune_refinement_limit = 3;

	fgac_contexti* pCtx;
	CUDA_VARIFY(cudaMalloc((void**)&pCtx, sizeof(fgac_contexti)));
	CUDA_VARIFY(cudaMemcpy(pCtx, &ctx, sizeof(fgac_contexti), cudaMemcpyHostToDevice));

	uint32_t texSize = dim_x * dim_y * 4 * sizeof(uint8_t);
	float* destData = nullptr;
	CUDA_VARIFY(cudaMalloc((void**)&destData, texSize));

	GPUDecodeKernel(dimGrid, dimBlock, buffer, (uint8_t*)destData, pCtx);

	CUDA_VARIFY(cudaDeviceSynchronize());
	float* hOutputData = (float*)malloc(texSize);
	CUDA_VARIFY(cudaMemcpy(hOutputData, destData, texSize, cudaMemcpyDeviceToHost));

	std::string outImagePath("G:/fgac/build/otest.tga");
	stbi_write_tga(outImagePath.c_str(), dim_x, dim_y, 4, hOutputData);
}

void CudaTestFunc()
{
	CudaTestCPUVersion();
	//CUDA_VARIFY(cudaSetDevice(0));
	//EncodeTest();
}
