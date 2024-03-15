#ifndef _FGAC_COMPRESS_SYMBOLIC_H_
#define _FGAC_COMPRESS_SYMBOLIC_H_
#include "fgac_decvice_common.cuh"
#include "fgac_ideal_endpoints_and_weights.cuh"
#include "../common/helper_math.h"

__device__ float compress_symbolic_block_for_partition_1plane(
	const block_size_descriptor& bsd,
	const image_block& blk,
	float tune_errorval_threshold,
	symbolic_compressed_block& scb,
	compression_working_buffers& tmpbuf,
	int quant_limit)
{
	int max_weight_quant = min(static_cast<int>(QUANT_32), quant_limit);

	// Compute ideal weights and endpoint colors, with no quantization or decimation
	endpoints_and_weights& ei = tmpbuf.ei1;
	compute_ideal_colors_and_weights_1plane(blk, ei);

	float* qwt_errors = tmpbuf.qwt_errors;
	unsigned int max_block_modes = bsd.block_mode_count_1plane_selected;
	for (uint32_t i = 0; i < max_block_modes; i++)
	{
		const block_mode& bm = bsd.block_modes[i];
		if (bm.quant_mode > max_weight_quant)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}

		int bitcount = 115 - 4 - bm.weight_bits;
		if (bitcount <= 0)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}
		float weights_uquantf[BLOCK_MAX_WEIGHTS];;
		compute_quantized_weights_for_decimation();
		qwt_errors[i] = compute_error_of_weight_set_1plane(ei, bsd,);
		
	}

	//{
	//	float errorval = ;
	//	if (errorval < best_errorval_in_scb)
	//	{
	//		best_errorval_in_scb = errorval;
	//		workscb.errorval = errorval;
	//		scb = workscb;
	//
	//		if (errorval < tune_errorval_threshold)
	//		{
	//			// Skip remaining candidates - this is "good enough"
	//			i = candidate_count;
	//			break;
	//		}
	//	}
	//}
	
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

	symbolic_compressed_block scb;

	// search 1 partition and 1 plane
	float errorval = compress_symbolic_block_for_partition_1plane(bsd, *blk, error_threshold, scb, *tmpbuf, QUANT_32);
}
#endif