#ifndef _FGAC_COMPRESS_SYMBOLIC_CUH_
#define _FGAC_COMPRESS_SYMBOLIC_CUH_

//todo: constexpt start_trial

#include "fgac_internal.cuh"
#include "fgac_pick_best_endpoint_format.cuh"
#include "fgac_ideal_endpoints_and_weights.cuh"
#include "fgac_weight_align.cuh"

__device__ float compress_symbolic_block_for_partition_1plane(const fgac_config& config,
	const block_size_descriptor& bsd,
	const image_block& blk,
	bool only_always,
	float tune_errorval_threshold,
	unsigned int partition_count,
	unsigned int partition_index,
	symbolic_compressed_block& scb,
	compression_working_buffers& tmpbuf,
	int quant_limit)
{
	int max_weight_quant = std::min(static_cast<int>(QUANT_32), quant_limit);

	// Compute ideal weights and endpoint colors, with no quantization or decimation
	const partition_info* pi = get_partition_info(&bsd,partition_count, partition_index);

	// Compute ideal weights and endpoint colors, with no quantization or decimation
	endpoints_and_weights& ei = tmpbuf.ei1;
	compute_ideal_colors_and_weights_1plane(blk, *pi, ei);

	// Compute ideal weights and endpoint colors for every decimation
	float* dec_weights_ideal = tmpbuf.dec_weights_ideal;
	uint8_t* dec_weights_uquant = tmpbuf.dec_weights_uquant;

	// For each decimation mode, compute an ideal set of weights with no quantization
	unsigned int max_decimation_modes = only_always ? bsd.decimation_mode_count_always
		: bsd.decimation_mode_count_selected;

	for (unsigned int i = 0; i < max_decimation_modes; i++)
	{
		const decimation_mode& dm = get_decimation_mode(&bsd, i);
		if (!is_ref_1plane(dm, static_cast<quant_method>(max_weight_quant)))
		{
			continue;
		}

		const decimation_info& di = get_decimation_info(&bsd, i);
		compute_ideal_weights_for_decimation(
			ei,
			di,
			dec_weights_ideal + i * BLOCK_MAX_WEIGHTS);
	}

	// Compute maximum colors for the endpoints and ideal weights, then for each endpoint and ideal
	// weight pair, compute the smallest weight that will result in a color value greater than 1

		// For each mode, use the angular method to compute a shift
	compute_angular_endpoints_1plane(
		only_always, bsd, dec_weights_ideal, max_weight_quant, tmpbuf);


	// For each mode (which specifies a decimation and a quantization):
	//     * Compute number of bits needed for the quantized weights
	//     * Generate an optimized set of quantized weights
	//     * Compute quantization errors for the mode
	// 

	// find quant mode
	uint32_t max_block_modes = ;
	for (uint32_t i = 0; i < max_block_modes; i++)
	{
	
	}
	
	uint32_t candidate_count = compute_ideal_endpoint_formats();
	
	// find best color end points
	// candidate_count
	for (unsigned int i = 0; i < candidate_count; i++)
	{
		// iterative time
		for (unsigned int l = 0; l < config.tune_refinement_limit; l++)
		{
			for (unsigned int j = 0; j < partition_count; j++)
			{
				// pack_color_endpoints
			}
	
			// If all the color endpoint modes are the same, we get a few more bits to store colors
			if (partition_count >= 2 && all_same)
			{
	
			}
	
			// compute_difference(difference between origianl color and compressed color)
		}
	}
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
	compress_symbolic_block_for_partition_1plane(1);
	return;
}

#endif