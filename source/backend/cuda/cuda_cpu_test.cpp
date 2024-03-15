#include <stdio.h>
#include <string>
#include <fstream>
#include <assert.h>

#include "fgac_cuda.h"
#include "fgac_compress_texture.h"
#include "fgac_internal.h"

#include "stb_image.h"
#include "stb_image_write.h"

#define __device__
#define __global__

#include "fgac_compress_symbolic.cuh"
#include "common\helper_math.h"

#include <iostream>

void CudaTestCPUVersion()
{
	std::string imagePath("G:/fgac/tex_test_4_4.png");
	int width = 0, height = 0, comp = 0;
	stbi_uc* srcData = stbi_load(imagePath.c_str(), &width, &height, &comp, STBI_rgb_alpha);
	uint32_t texSize = width * height * 4 * sizeof(uint8_t);

	fgac_contexti* ctx = new fgac_contexti();
	ctx->dim_x = width;
	ctx->dim_y = height;
	ctx->bsd.xdim = 4;
	ctx->bsd.ydim = 4;
	ctx->bsd.texel_count = 4 * 4;
	ctx->config.cw_r_weight = 1;
	ctx->config.cw_g_weight = 1;
	ctx->config.cw_b_weight = 1;
	ctx->config.cw_a_weight = 1;
	ctx->config.cw_sum_weight = 4;
	ctx->config.tune_db_limit = 2254.6614;
	ctx->config.tune_candidate_limit = 3;
	ctx->config.tune_refinement_limit = 3;
	
	uint8_t pcb[16];
	compression_working_buffers* buffer = new compression_working_buffers();
	image_block img_blk;

	float4 data_min(1e20, 1e20, 1e20, 1e20);
	float4 data_max(0, 0, 0, 0);
	float4 data_mean(0, 0, 0, 0);

	int idx = 0;
	for (uint32_t index_y = 0; index_y < 4; index_y++)
	{
		for (uint32_t index_x = 0; index_x < 4; index_x++)
		{
			uint8_t* data = &srcData[(index_y * 4 + index_x) * 4];
			float4 datav = make_float4(data[0], data[1], data[2], data[3]) * (65535.0f / 255.0f);


			data_min = fminf(datav, data_min);
			data_max = fmaxf(datav, data_max);
			data_mean += datav;

			img_blk.data_r[idx] = datav.x;
			img_blk.data_g[idx] = datav.y;
			img_blk.data_b[idx] = datav.z;
			img_blk.data_a[idx] = datav.w;

			std::cout
				<< "data r:" << img_blk.data_r[idx]
				<< " data g:" << img_blk.data_g[idx]
				<< " data b:" << img_blk.data_b[idx]
				<< " data a:" << img_blk.data_a[idx] << std::endl;


			idx++;
		}
	}

	img_blk.data_min = data_min;
	img_blk.data_max = data_max;
	img_blk.data_mean = data_mean / static_cast<float>(ctx->bsd.texel_count);
	img_blk.channel_weight = float4(
		ctx->config.cw_r_weight,
		ctx->config.cw_g_weight,
		ctx->config.cw_b_weight,
		ctx->config.cw_a_weight);

	init_block_size_descriptor(4, 4, 1, true, 1, 0, ctx->bsd);
	compress_block(ctx, &img_blk, pcb, buffer);

}