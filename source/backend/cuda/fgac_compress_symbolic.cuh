#ifndef _FGAC_COMPRESS_SYMBOLIC_CUH_
#define _FGAC_COMPRESS_SYMBOLIC_CUH_

//todo: constexpt start_trial

#include "fgac_internal.cuh"
//#include "fgac_pick_best_endpoint_format.cuh"
//#include "fgac_ideal_endpoints_and_weights.cuh"
//#include "fgac_weight_align.cuh"
//#include "fgac_color_quantize.cuh"
//#include "fgac_decompress_symbolic.cuh"
//#include "fgac_symbolic_physical.cuh"

//__constant__ int8_t free_bits_for_partition_count[4]{
//	115 - 4, 111 - 4 - PARTITION_INDEX_BITS, 108 - 4 - PARTITION_INDEX_BITS, 105 - 4 - PARTITION_INDEX_BITS
//};
//
__device__ float compress_symbolic_block_for_partition_1plane(const fgac_config& config,
	const block_size_descriptor& bsd,
	const image_block& blk,
	float tune_errorval_threshold,
	symbolic_compressed_block& scb,
	compression_working_buffers& tmpbuf,
	int quant_limit)
{
	int max_weight_quant = std::min(static_cast<int>(QUANT_32), quant_limit);

	// Compute ideal weights and endpoint colors, with no quantization or decimation
	const partition_info* pi = nullptr;
//
//	// Compute ideal weights and endpoint colors, with no quantization or decimation
//	endpoints_and_weights& ei = tmpbuf.ei1;
//	compute_ideal_colors_and_weights_1plane(blk, *pi, ei);
//
//	// Compute ideal weights and endpoint colors for every decimation
//	float* dec_weights_ideal = tmpbuf.dec_weights_ideal;
//	uint8_t* dec_weights_uquant = tmpbuf.dec_weights_uquant;
//
//	// For each decimation mode, compute an ideal set of weights with no quantization
//	unsigned int max_decimation_modes = only_always ? bsd.decimation_mode_count_always
//		: bsd.decimation_mode_count_selected;
//
//	for (unsigned int i = 0; i < max_decimation_modes; i++)
//	{
//		const decimation_mode& dm = get_decimation_mode(&bsd, i);
//		if (!is_ref_1plane(dm, static_cast<quant_method>(max_weight_quant)))
//		{
//			continue;
//		}
//
//		const decimation_info& di = get_decimation_info(&bsd, i);
//		compute_ideal_weights_for_decimation(
//			ei,
//			di,
//			dec_weights_ideal + i * BLOCK_MAX_WEIGHTS);
//	}
//
//	// Compute maximum colors for the endpoints and ideal weights, then for each endpoint and ideal
//	// weight pair, compute the smallest weight that will result in a color value greater than 1
//	float4 min_ep(10.0f);
//	for (unsigned int i = 0; i < partition_count; i++)
//	{
//		float4 ep = (float4(1.0f) - ei.ep.endpt0[i]) / (ei.ep.endpt1[i] - ei.ep.endpt0[i]);
//
//		min_ep = float4(
//			(ep.x > 0.5 && ep.x < min_ep.x) ? ep.x : min_ep.x,
//			(ep.y > 0.5 && ep.y < min_ep.y) ? ep.y : min_ep.y,
//			(ep.z > 0.5 && ep.z < min_ep.z) ? ep.z : min_ep.z,
//			(ep.w > 0.5 && ep.w < min_ep.w) ? ep.w : min_ep.w);
//	}
//
//	float min_wt_cutoff = std::min(std::min(min_ep.x, min_ep.y), std::min(min_ep.z, min_ep.w));
//
//	// For each mode, use the angular method to compute a shift
//	compute_angular_endpoints_1plane(
//		only_always, bsd, dec_weights_ideal, max_weight_quant, tmpbuf);
//
//	float* weight_low_value = tmpbuf.weight_low_value1;
//	float* weight_high_value = tmpbuf.weight_high_value1;
//	int8_t* qwt_bitcounts = tmpbuf.qwt_bitcounts;
//	float* qwt_errors = tmpbuf.qwt_errors;
//
//	// For each mode (which specifies a decimation and a quantization):
//	//     * Compute number of bits needed for the quantized weights
//	//     * Generate an optimized set of quantized weights
//	//     * Compute quantization errors for the mode
//	// 
//
//	unsigned int max_block_modes = only_always ? bsd.block_mode_count_1plane_always
//		: bsd.block_mode_count_1plane_selected;
//
//	// find quant mode
//	for (uint32_t i = 0; i < max_block_modes; i++)
//	{
//		const block_mode& bm = bsd.block_modes[i];
//		if (bm.quant_mode > max_weight_quant)
//		{
//			qwt_errors[i] = 1e38f;
//			continue;
//		}
//
//		int bitcount = free_bits_for_partition_count[partition_count - 1] - bm.weight_bits;
//		if (bitcount <= 0)
//		{
//			qwt_errors[i] = 1e38f;
//			continue;
//		}
//
//		if (weight_high_value[i] > 1.02f * min_wt_cutoff)
//		{
//			weight_high_value[i] = 1.0f;
//		}
//
//		int decimation_mode = bm.decimation_mode;
//		const decimation_info& di = get_decimation_info(&bsd, decimation_mode);
//
//		qwt_bitcounts[i] = static_cast<int8_t>(bitcount);
//		
//		float dec_weights_uquantf[BLOCK_MAX_WEIGHTS];
//
//		// Generate the optimized set of weights for the weight mode
//		compute_quantized_weights_for_decimation(
//			bsd,
//			di,
//			weight_low_value[i], weight_high_value[i],
//			dec_weights_ideal + BLOCK_MAX_WEIGHTS * decimation_mode,
//			dec_weights_uquantf,
//			dec_weights_uquant + BLOCK_MAX_WEIGHTS * i,
//			bm.get_weight_quant_mode());
//
//		// Compute weight quantization errors for the block mode
//		qwt_errors[i] = compute_error_of_weight_set_1plane(
//			ei,
//			di,
//			dec_weights_uquantf);
//	}
//	
//	// Decide the optimal combination of color endpoint encodings and weight encodings
//	uint8_t partition_format_specifiers[TUNE_MAX_TRIAL_CANDIDATES][BLOCK_MAX_PARTITIONS];
//	int block_mode_index[TUNE_MAX_TRIAL_CANDIDATES];
//
//	quant_method color_quant_level[TUNE_MAX_TRIAL_CANDIDATES];
//	quant_method color_quant_level_mod[TUNE_MAX_TRIAL_CANDIDATES];
//
//	// Iterate over the N believed-to-be-best modes to find out which one is actually best
	float best_errorval_in_mode = ERROR_CALC_DEFAULT;
//	float best_errorval_in_scb = scb.errorval;
//
//	uint32_t candidate_count = compute_ideal_endpoint_formats(
//		bsd, *pi, blk, ei.ep, qwt_bitcounts, qwt_errors,
//		config.tune_candidate_limit, 0, max_block_modes,
//		partition_format_specifiers, block_mode_index,
//		color_quant_level, color_quant_level_mod, tmpbuf);
//	
//	// Iterate over the N believed-to-be-best modes to find out which one is actually best
//	for (unsigned int i = 0; i < candidate_count; i++)
//	{
//		const int bm_packed_index = block_mode_index[i];
//		const block_mode& qw_bm = bsd.block_modes[bm_packed_index];
//
//		int decimation_mode = qw_bm.decimation_mode;
//		const decimation_info& di = get_decimation_info(&bsd,decimation_mode);
//
//		// Recompute the ideal color endpoints before storing them
//		float4 rgbs_colors[BLOCK_MAX_PARTITIONS];
//		float4 rgbo_colors[BLOCK_MAX_PARTITIONS];
//
//		symbolic_compressed_block workscb;
//		endpoints workep = ei.ep;
//
//		uint8_t* u8_weight_src = dec_weights_uquant + BLOCK_MAX_WEIGHTS * bm_packed_index;
//
//		for (unsigned int j = 0; j < di.weight_count; j++)
//		{
//			workscb.weights[j] = u8_weight_src[j];
//		}
//
//		for (unsigned int l = 0; l < config.tune_refinement_limit; l++)
//		{
//			recompute_ideal_colors_1plane(
//				blk, *pi, di, workscb.weights,
//				workep, rgbs_colors, rgbo_colors);
//
//			// Quantize the chosen color, tracking if worth trying the mod value
//			bool all_same = color_quant_level[i] != color_quant_level_mod[i];
//			for (unsigned int j = 0; j < partition_count; j++)
//			{
//				workscb.color_formats[j] = pack_color_endpoints(
//					bsd,
//					workep.endpt0[j],
//					workep.endpt1[j],
//					rgbs_colors[j],
//					rgbo_colors[j],
//					partition_format_specifiers[i][j],
//					workscb.color_values[j],
//					color_quant_level[i]);
//
//				all_same = all_same && workscb.color_formats[j] == workscb.color_formats[0];
//			}
//
//			// If all the color endpoint modes are the same, we get a few more bits to store colors;
//			// let's see if we can take advantage of this: requantize all the colors and see if the
//			// endpoint modes remain the same.
//			workscb.color_formats_matched = 0;
//			if (partition_count >= 2 && all_same)
//			{
//				uint8_t colorvals[BLOCK_MAX_PARTITIONS][8];
//				uint8_t color_formats_mod[BLOCK_MAX_PARTITIONS]{ 0 };
//				bool all_same_mod = true;
//				for (unsigned int j = 0; j < partition_count; j++)
//				{
//					color_formats_mod[j] = pack_color_endpoints(
//						bsd,
//						workep.endpt0[j],
//						workep.endpt1[j],
//						rgbs_colors[j],
//						rgbo_colors[j],
//						partition_format_specifiers[i][j],
//						colorvals[j],
//						color_quant_level_mod[i]);
//
//					// Early out as soon as it's no longer possible to use mod
//					if (color_formats_mod[j] != color_formats_mod[0])
//					{
//						all_same_mod = false;
//						break;
//					}
//				}
//
//				if (all_same_mod)
//				{
//					workscb.color_formats_matched = 1;
//					for (unsigned int j = 0; j < BLOCK_MAX_PARTITIONS; j++)
//					{
//						for (unsigned int k = 0; k < 8; k++)
//						{
//							workscb.color_values[j][k] = colorvals[j][k];
//						}
//
//						workscb.color_formats[j] = color_formats_mod[j];
//					}
//				}
//			}
//
//			// Store header fields
//			workscb.partition_count = static_cast<uint8_t>(partition_count);
//			workscb.partition_index = static_cast<uint16_t>(partition_index);
//			workscb.plane2_component = -1;
//			workscb.quant_mode = workscb.color_formats_matched ? color_quant_level_mod[i] : color_quant_level[i];
//			workscb.block_mode = qw_bm.mode_index;
//			workscb.block_type = SYM_BTYPE_NONCONST;
//
//			// Pre-realign test
//			if (l == 0)
//			{
//				float errorval = compute_symbolic_block_difference_1plane_1partition(config, bsd, workscb, blk);
//				if (errorval == -ERROR_CALC_DEFAULT)
//				{
//					errorval = -errorval;
//					workscb.block_type = SYM_BTYPE_ERROR;
//				}
//
//				best_errorval_in_mode = fmin(errorval, best_errorval_in_mode);
//
//				// Average refinement improvement is 3.5% per iteration (allow 4.5%), but the first
//				// iteration can help more so we give it a extra 8% leeway. Use this knowledge to
//				// drive a heuristic to skip blocks that are unlikely to catch up with the best
//				// block we have already.
//				unsigned int iters_remaining = config.tune_refinement_limit - l;
//				float threshold = (0.045f * static_cast<float>(iters_remaining)) + 1.08f;
//				if (errorval > (threshold * best_errorval_in_scb))
//				{
//					break;
//				}
//
//				if (errorval < best_errorval_in_scb)
//				{
//					best_errorval_in_scb = errorval;
//					workscb.errorval = errorval;
//					scb = workscb;
//
//					if (errorval < tune_errorval_threshold)
//					{
//						// Skip remaining candidates - this is "good enough"
//						i = candidate_count;
//						break;
//					}
//				}
//			}
//
//			// Post-realign test
//			float errorval = compute_symbolic_block_difference_1plane_1partition(config, bsd, workscb, blk);
//			if (errorval == -ERROR_CALC_DEFAULT)
//			{
//				errorval = -errorval;
//				workscb.block_type = SYM_BTYPE_ERROR;
//			}
//
//			best_errorval_in_mode = fmin(errorval, best_errorval_in_mode);
//
//			// Average refinement improvement is 3.5% per iteration, so skip blocks that are
//			// unlikely to catch up with the best block we have already. Assume a 4.5% per step to
//			// give benefit of the doubt ...
//			unsigned int iters_remaining = config.tune_refinement_limit - 1 - l;
//			float threshold = (0.045f * static_cast<float>(iters_remaining)) + 1.0f;
//			if (errorval > (threshold * best_errorval_in_scb))
//			{
//				break;
//			}
//
//			if (errorval < best_errorval_in_scb)
//			{
//				best_errorval_in_scb = errorval;
//				workscb.errorval = errorval;
//				scb = workscb;
//
//				if (errorval < tune_errorval_threshold)
//				{
//					// Skip remaining candidates - this is "good enough"
//					i = candidate_count;
//					break;
//				}
//			}
//
//			
//		}
//	}
//
	return best_errorval_in_mode;
}

__device__ void compress_block(fgac_contexti* ctx, image_block* blk, uint8_t pcb[16], compression_working_buffers* tmpbuf)
{
	const block_size_descriptor& bsd = ctx->bsd;
	symbolic_compressed_block scb;
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
	{
		float errorval = compress_symbolic_block_for_partition_1plane(
			ctx->config, bsd, *blk, error_threshold, scb, *tmpbuf, QUANT_32);
	}
	
	// Compress to a physical block
	//symbolic_to_physical(bsd, scb, pcb);
}

#endif