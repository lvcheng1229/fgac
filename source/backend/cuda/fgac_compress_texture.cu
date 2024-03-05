#ifndef _FGAC_COMPRESS_TEXTURE_CU_
#define _FGAC_COMPRESS_TEXTURE_CU_


#include "fgac_compress_texture.h"
#include "fgac_compress_texture.cuh"

// note:
// 1. don't support hdr compression for now
// 2. 

__device__ uchar4 GetTexelData(uint2 blockStartIndex, float2 invTexSize, uint32_t subBlokcIndex, cudaTextureObject_t tex)
{
	uint32_t subIndexY = subBlokcIndex / BLOCK_DIM_WIDTH;
	uint32_t subIndexX = subBlokcIndex - subIndexY *  BLOCK_DIM_WIDTH;

	uint2 sampledIndexXY = uint2(blockStartIndex.x + subIndexX, blockStartIndex.y + subIndexY);
	float2 sampledUV = float2(sampledIndexXY.x * invTexSize.x + sampledIndexXY.y * invTexSize.y);
	return tex2D<uchar4>(tex, sampledUV.x, sampledUV.y);
}

__device__ void compress_symbolic_block(imageblock* blk, symbolic_compressed_block* scb, SAstcEncoderInfo* astcEncoderInfo)
{
	if (blk->red_min == blk->red_max && blk->green_min == blk->green_max && blk->blue_min == blk->blue_max && blk->alpha_min == blk->alpha_max)
	{
		// Encode as UNORM16 if NOT using HDR.
		scb->block_mode = -2;
		scb->partition_count = 0;
		float red = blk->orig_data[0].x;
		float green = blk->orig_data[0].y;
		float blue = blk->orig_data[0].z;
		float alpha = blk->orig_data[0].w;
		if (red < 0)
			red = 0;
		else if (red > 1)
			red = 1;
		if (green < 0)
			green = 0;
		else if (green > 1)
			green = 1;
		if (blue < 0)
			blue = 0;
		else if (blue > 1)
			blue = 1;
		if (alpha < 0)
			alpha = 0;
		else if (alpha > 1)
			alpha = 1;
		scb->constant_color[0] = (int)floor(red * 65535.0f + 0.5f);
		scb->constant_color[1] = (int)floor(green * 65535.0f + 0.5f);
		scb->constant_color[2] = (int)floor(blue * 65535.0f + 0.5f);
		scb->constant_color[3] = (int)floor(alpha * 65535.0f + 0.5f);
		return;
	}
}

__global__ void GPUEncodeKernel(uint8_t* outputData, cudaTextureObject_t tex, SAstcEncoderInfo* astcEncoderInfo)
{
	// calculate normalized texture coordinates
	uint32_t globalIndexX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t globalIndexY = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (blockIdx.x >= astcEncoderInfo->m_blkNumX || blockIdx.y >= astcEncoderInfo->m_blkNumY)
	{
		return;
	}

	uint2 blockStartIndex = make_uint2(globalIndexX, globalIndexY);
	float2 invTexSize = make_float2(1.0f / astcEncoderInfo->m_srcTexWidth, 1.0 / astcEncoderInfo->m_srcTexHeight);

	imageblock blk;
	uchar4 blockImage[BLOCK_PIXEL_NUM];

	float red_min = FLOAT_38, red_max = -FLOAT_38;
	float green_min = FLOAT_38, green_max = -FLOAT_38;
	float blue_min = FLOAT_38, blue_max = -FLOAT_38;
	float alpha_min = FLOAT_38, alpha_max = -FLOAT_38;

	for (uint32_t subBlockIndex = 0; subBlockIndex < BLOCK_PIXEL_NUM; subBlockIndex++)
	{
		uchar4 texelData = GetTexelData(blockStartIndex, invTexSize, subBlockIndex, tex);
		blockImage[subBlockIndex] = texelData;
		
		float red = texelData.x / 255.0;
		float blue = texelData.y / 255.0;
		float green = texelData.z / 255.0;
		float alpha = texelData.w / 255.0;

		if (red < red_min)
			red_min = red;
		if (red > red_max)
			red_max = red;
		if (green < green_min)
			green_min = green;
		if (green > green_max)
			green_max = green;
		if (blue < blue_min)
			blue_min = blue;
		if (blue > blue_max)
			blue_max = blue;
		if (alpha < alpha_min)
			alpha_min = alpha;
		if (alpha > alpha_max)
			alpha_max = alpha;

		blk.orig_data[subBlockIndex] = make_float4(red, green, blue, alpha);
	}

	symbolic_compressed_block scb;
	compress_symbolic_block(&blk,&scb, astcEncoderInfo);

	uint32_t destBlockIndex = (globalIndexY * astcEncoderInfo->m_blkNumX + globalIndexX) * BYTES_PER_DESTINATION_BLOCK;
	
	outputData[destBlockIndex] = uint32_t(blockImage[0].x);
}

extern "C" void GPUEncodeKernel(dim3 gridSize, dim3 blockSize, uint8_t * outputData, cudaTextureObject_t tex, SAstcEncoderInfo * astcEncoderInfo)
{
	GPUEncodeKernel << <gridSize, blockSize, 0 >> > (outputData,tex,astcEncoderInfo);
}

#endif