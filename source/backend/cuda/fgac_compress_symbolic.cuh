#ifndef _FGAC_COMPRESS_SYMBOLIC_CUH_
#define _FGAC_COMPRESS_SYMBOLIC_CUH_

//todo: constexpt start_trial

#include "fgac_internal.cuh"

__device__ void compress_block(fgac_contexti* ctx, image_block* blk, uint8_t pcb[16], compression_working_buffers* tmpbuf)
{
	const block_size_descriptor& bsd = ctx->bsd;
	const fgac_config& config = ctx->config;

	if ((blk->data_min.x == blk->data_max.x) && 
		(blk->data_min.y == blk->data_max.y) && 
		(blk->data_min.z == blk->data_max.z) && 
		(blk->data_min.w == blk->data_max.w))
	{
		//todo:
	}

	float error_weight_sum = config.cw_sum_weight * bsd.texel_count;
	float error_threshold = config.tune_db_limit * error_weight_sum;

	// use fast path?
	int start_trial = 1;
	if (config.tune_search_mode0_enable >= TUNE_MIN_SEARCH_MODE0)
	{
		start_trial = 0;
	}

	for (int i = start_trial; i < 2; i++)
	{

	}
	return;
}

#endif