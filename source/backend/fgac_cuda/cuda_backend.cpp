#include <stdio.h>
#include <string>
#include <fstream>
#include <assert.h>

#include <vector_types.h>

#include "stb_image.h"
#include "stb_image_write.h"

#include "cuda_backend.h"
#include "fgac_device_host_common.h"
#include "host\fgac_host_common.h"

#include "common\helper_math.h"

#define __device__
#define __global__
#include "device\fgac_decvice_common.cuh"
#include "device\fgac_compress_symbolic.cuh"

void cuda_backend_test()
{
	static constexpr uint32_t blockx = 4;
	static constexpr uint32_t blocky = 4;

	std::string imagePath("G:/fgac/tex_test_4_4.png");
	int width = 0, height = 0, comp = 0;
	stbi_uc* srcData = stbi_load(imagePath.c_str(), &width, &height, &comp, STBI_rgb_alpha);
	uint32_t texSize = width * height * 4 * sizeof(uint8_t);

	fgac_contexti* ctx = new fgac_contexti();
	ctx->config.cw_sum_weight = 4;
	ctx->config.tune_db_limit = 2254.6614;
	
	//tobe init
	//block_mode_count_1plane_selected
	
	init_block_size_descriptor(blockx, blocky, ctx->bsd);

	float4 data_min(1e20, 1e20, 1e20, 1e20);
	float4 data_max(0, 0, 0, 0);
	float4 data_mean(0, 0, 0, 0);

	image_block img_blk;

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

			idx++;
		}
	}

	img_blk.data_min = data_min;
	img_blk.data_max = data_max;
	img_blk.data_mean = data_mean / static_cast<float>(ctx->bsd.texel_count);

	uint8_t pcb[16];
	compression_working_buffers tmpbuf;
	compress_block(ctx, &img_blk, pcb, &tmpbuf);
}