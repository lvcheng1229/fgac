#ifndef _FGAC_COMPRESS_SYMBOLIC_CUH_
#define _FGAC_COMPRESS_SYMBOLIC_CUH_

//todo: constexpt start_trial

#include "fgac_internal.cuh"
#include "fgac_pick_best_endpoint_format.cuh"

__device__ float compress_symbolic_block_for_multi_partition_1plane(uint32_t partition_count)
{
	// For each mode (which specifies a decimation and a quantization):
	//     * Compute number of bits needed for the quantized weights
	//     * Generate an optimized set of quantized weights
	//     * Compute quantization errors for the mode

	// find quant mode
	//	uint32_t max_block_modes = ;
	//	for (uint32_t i = 0; i < max_block_modes; i++)
	//	{
	//	
	//	}
	//	
	//	uint32_t candidate_count = compute_ideal_endpoint_formats();
	//	
	//	// find best color end points
	//	// candidate_count
	//	for (unsigned int i = 0; i < candidate_count; i++)
	//	{
	//		// iterative time
	//		for (unsigned int l = 0; l < config.tune_refinement_limit; l++)
	//		{
	//			for (unsigned int j = 0; j < partition_count; j++)
	//			{
	//				// pack_color_endpoints
	//			}
	//	
	//			// If all the color endpoint modes are the same, we get a few more bits to store colors
	//			if (partition_count >= 2 && all_same)
	//			{
	//	
	//			}
	//	
	//			// compute_difference(difference between origianl color and compressed color)
	//		}
	//	}
}

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

	// search 1 partition and 1 plane
	compress_symbolic_block_for_multi_partition_1plane(1);
	return;
}

#endif