#ifndef __FGAC_IMAGE_CUH_
#define __FGAC_IMAGE_CUH_
#include "fgac_internal.cuh"
#include "common\helper_math.h"

__device__ void load_image_block_fast_ldr(image_block* blk, uint2 startPixPos, cudaTextureObject_t tex, fgac_contexti* ctx)
{
	const block_size_descriptor bsd = ctx->bsd;
	const float2 invTexSize = make_float2(1.0 / ctx->dim_x, 1.0 / ctx->dim_y);
	
	float4 data_min(1e38);
	float4 data_max(-1e38);
	float4 data_mean(0);
	
	int idx = 0;
#if CUDA_DEBUG
	for (uint32_t y = 0; y < 4; y++)
	{
		for (uint32_t x = 0; x < 4; x++)
		{
#else
	for (uint32_t y = 0; y < bsd.ydim; y++)
	{
		for (uint32_t x = 0; x < bsd.xdim; x++)
		{
#endif
			const uint2 sampledIndexXY = uint2(startPixPos.x + x, startPixPos.y + y);
			const float2 sampledUV = make_float2(sampledIndexXY.x * invTexSize.x, sampledIndexXY.y * invTexSize.y);
			uchar4 srcData = tex2D<uchar4>(tex, sampledUV.x, sampledUV.y);
	
			float4 datav = make_float4(srcData.x, srcData.y, srcData.z, srcData.w) * (65535.0f / 255.0f);
			
			data_min = fminf(datav, data_min);
			data_max = fmaxf(datav, data_max);
			data_mean += datav;
	
			blk->data_r[idx] = datav.x;
			blk->data_g[idx] = datav.y;
			blk->data_b[idx] = datav.z;
			blk->data_a[idx] = datav.w;

			idx++;
		}
	}

	blk->data_min = data_min;
	blk->data_max = data_max;
	blk->data_mean = data_mean / static_cast<float>(bsd.texel_count);
} 

#endif
