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

	int idx = 0;
	for (uint32_t y = 0; y < bsd.ydim; y++)
	{
		for (uint32_t x = 0; x < bsd.xdim; x++)
		{
			const uint2 sampledIndexXY = uint2(startPixPos.x + x, startPixPos.y + y);
			const float2 sampledUV = make_float2(sampledIndexXY.x * invTexSize.x, sampledIndexXY.y * invTexSize.y);
			uchar4 srcData = tex2D<uchar4>(tex, sampledUV.x, sampledUV.y);

			float4 datav = make_float4(srcData.x, srcData.y, srcData.z, srcData.w) * (65535.0f / 255.0f);
			
			data_min = fminf(datav, data_min);
			data_max = fmaxf(datav, data_max);

			blk->data_r[idx] = srcData.x;
			blk->data_g[idx] = srcData.y;
			blk->data_b[idx] = srcData.z;
			blk->data_a[idx] = srcData.w;
			
			idx++;
		}
	}

	blk->data_min = data_min;
	blk->data_max = data_max;
} 

#endif
