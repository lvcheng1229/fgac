#ifndef _FGAC_COMPRESS_SYMBOLIC_CUH_
#define _FGAC_COMPRESS_SYMBOLIC_CUH_

//todo: constexpt start_trial

#include "fgac_internal.cuh"
#include "fgac_pick_best_endpoint_format.cuh"
#include "fgac_ideal_endpoints_and_weights.cuh"
#include "fgac_weight_align.cuh"

__constant__ int8_t free_bits_for_partition_count[4]{
	115 - 4, 111 - 4 - PARTITION_INDEX_BITS, 108 - 4 - PARTITION_INDEX_BITS, 105 - 4 - PARTITION_INDEX_BITS
};

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
	float4 min_ep(10.0f);
	for (unsigned int i = 0; i < partition_count; i++)
	{
		float4 ep = (float4(1.0f) - ei.ep.endpt0[i]) / (ei.ep.endpt1[i] - ei.ep.endpt0[i]);

		min_ep = float4(
			(ep.x > 0.5 && ep.x < min_ep.x) ? ep.x : min_ep.x,
			(ep.y > 0.5 && ep.y < min_ep.y) ? ep.y : min_ep.y,
			(ep.z > 0.5 && ep.z < min_ep.z) ? ep.z : min_ep.z,
			(ep.w > 0.5 && ep.w < min_ep.w) ? ep.w : min_ep.w);
	}

	float min_wt_cutoff = std::min(std::min(min_ep.x, min_ep.y), std::min(min_ep.z, min_ep.w));

	// For each mode, use the angular method to compute a shift
	compute_angular_endpoints_1plane(
		only_always, bsd, dec_weights_ideal, max_weight_quant, tmpbuf);

	float* weight_low_value = tmpbuf.weight_low_value1;
	float* weight_high_value = tmpbuf.weight_high_value1;
	int8_t* qwt_bitcounts = tmpbuf.qwt_bitcounts;
	float* qwt_errors = tmpbuf.qwt_errors;

	// For each mode (which specifies a decimation and a quantization):
	//     * Compute number of bits needed for the quantized weights
	//     * Generate an optimized set of quantized weights
	//     * Compute quantization errors for the mode
	// 

	unsigned int max_block_modes = only_always ? bsd.block_mode_count_1plane_always
		: bsd.block_mode_count_1plane_selected;

	// find quant mode
	for (uint32_t i = 0; i < max_block_modes; i++)
	{
		const block_mode& bm = bsd.block_modes[i];
		if (bm.quant_mode > max_weight_quant)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}

		int bitcount = free_bits_for_partition_count[partition_count - 1] - bm.weight_bits;
		if (bitcount <= 0)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}

		if (weight_high_value[i] > 1.02f * min_wt_cutoff)
		{
			weight_high_value[i] = 1.0f;
		}

		int decimation_mode = bm.decimation_mode;
		const decimation_info& di = get_decimation_info(&bsd, decimation_mode);

		qwt_bitcounts[i] = static_cast<int8_t>(bitcount);
		
		float dec_weights_uquantf[BLOCK_MAX_WEIGHTS];

		// Generate the optimized set of weights for the weight mode
		compute_quantized_weights_for_decimation(
			bsd,
			di,
			weight_low_value[i], weight_high_value[i],
			dec_weights_ideal + BLOCK_MAX_WEIGHTS * decimation_mode,
			dec_weights_uquantf,
			dec_weights_uquant + BLOCK_MAX_WEIGHTS * i,
			bm.get_weight_quant_mode());

		// Compute weight quantization errors for the block mode
		qwt_errors[i] = compute_error_of_weight_set_1plane(
			ei,
			di,
			dec_weights_uquantf);
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