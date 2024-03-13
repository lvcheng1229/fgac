#ifndef _FGAC_IDEAL_ENDPOINTS_AND_WEIGHTS_CUH_
#define _FGAC_IDEAL_ENDPOINTS_AND_WEIGHTS_CUH_
#include "fgac_internal.cuh"
#include "fgac_averages_and_directions.cuh"

//todo:is_direct

__device__ float bilinear_infill_vla(
	const decimation_info& di,
	const float* weights,
	unsigned int index
) {
	// Load the bilinear filter texel weight indexes in the decimated grid
	int weight_idx0 = int(di.texel_weights_tr[0] + index);
	int weight_idx1 = int(di.texel_weights_tr[1] + index);
	int weight_idx2 = int(di.texel_weights_tr[2] + index);
	int weight_idx3 = int(di.texel_weights_tr[3] + index);

	// Load the bilinear filter weights from the decimated grid
	float weight_val0 = weights[weight_idx0];
	float weight_val1 = weights[weight_idx1];
	float weight_val2 = weights[weight_idx2];
	float weight_val3 = weights[weight_idx3];

	// Load the weight contribution factors for each decimated weight
	float tex_weight_float0 = di.texel_weight_contribs_float_tr[0][index];
	float tex_weight_float1 = di.texel_weight_contribs_float_tr[1][index];
	float tex_weight_float2 = di.texel_weight_contribs_float_tr[2][index];
	float tex_weight_float3 = di.texel_weight_contribs_float_tr[3][index];

	// Compute the bilinear interpolation to generate the per-texel weight
	return (weight_val0 * tex_weight_float0 + weight_val1 * tex_weight_float1) +
		(weight_val2 * tex_weight_float2 + weight_val3 * tex_weight_float3);
}

__device__ float bilinear_infill_vla(
	const decimation_info& di,
	const float* weights,
	unsigned int index
) {
	// Load the bilinear filter texel weight indexes in the decimated grid
	int weight_idx0 = di.texel_weights_tr[0][index];
	int weight_idx1 = di.texel_weights_tr[1][index];
	int weight_idx2 = di.texel_weights_tr[2][index];
	int weight_idx3 = di.texel_weights_tr[3][index];

	// Load the bilinear filter weights from the decimated grid
	float weight_val0 = weights[weight_idx0];
	float weight_val1 = weights[weight_idx1];
	float weight_val2 = weights[weight_idx2];
	float weight_val3 = weights[weight_idx3];

	// Load the weight contribution factors for each decimated weight
	float tex_weight_float0 = di.texel_weight_contribs_float_tr[0][index];
	float tex_weight_float1 = di.texel_weight_contribs_float_tr[1][index];
	float tex_weight_float2 = di.texel_weight_contribs_float_tr[2][index];
	float tex_weight_float3 = di.texel_weight_contribs_float_tr[3][index];

	// Compute the bilinear interpolation to generate the per-texel weight
	return (weight_val0 * tex_weight_float0 + weight_val1 * tex_weight_float1) +
		(weight_val2 * tex_weight_float2 + weight_val3 * tex_weight_float3);
}

__device__ float bilinear_infill_vla_2(
	const decimation_info& di,
	const float* weights,
	unsigned int index
) {
	// Load the bilinear filter texel weight indexes in the decimated grid
	int weight_idx0 = di.texel_weights_tr[0][index];
	int weight_idx1 = di.texel_weights_tr[1][index];

	// Load the bilinear filter weights from the decimated grid
	float weight_val0 = weights[weight_idx0];
	float weight_val1 = weights[weight_idx1];

	// Load the weight contribution factors for each decimated weight
	float tex_weight_float0 = di.texel_weight_contribs_float_tr[0][index];
	float tex_weight_float1 = di.texel_weight_contribs_float_tr[1][index];

	// Compute the bilinear interpolation to generate the per-texel weight
	return (weight_val0 * tex_weight_float0 + weight_val1 * tex_weight_float1);
}

void compute_ideal_weights_for_decimation(
	const endpoints_and_weights& ei,
	const decimation_info& di,
	float* dec_weight_ideal_value
) 
{
	unsigned int texel_count = di.texel_count;
	unsigned int weight_count = di.weight_count;
	bool is_direct = texel_count == weight_count;
	
	if (is_direct)
	{
		for (unsigned int i = 0; i < texel_count; i++)
		{
			dec_weight_ideal_value[i] = ei.weights[i];
		}
		return;
	}

	// Otherwise compute an estimate and perform single refinement iteration
	float infilled_weights[BLOCK_MAX_TEXELS];

	// Compute an initial average for each decimated weight
	bool constant_wes = ei.is_constant_weight_error_scale;
	float weight_error_scale(ei.weight_error_scale[0]);

	for (unsigned int i = 0; i < weight_count; i ++)
	{
		// Start with a small value to avoid div-by-zero later
		float weight_weight(1e-10f);
		float initial_weight = 0;

		uint8_t weight_texel_count = di.weight_texel_count[i];
		for (unsigned int j = 0; j < weight_texel_count; j++)
		{
			const uint8_t texel_index = di.weight_texels_tr[j][i];
			float weight = di.weights_texel_contribs_tr[j][i];

			if (!constant_wes)
			{
				weight_error_scale = ei.weight_error_scale[texel_index];
			}

			float contrib_weight = weight * weight_error_scale;

			weight_weight += contrib_weight;
			initial_weight += ei.weights[texel_index] * contrib_weight;
		}

		dec_weight_ideal_value[i] = initial_weight / weight_weight;
	}

	if (di.max_texel_weight_count <= 2)
	{
		for (unsigned int i = 0; i < texel_count; i ++)
		{
			float weight = bilinear_infill_vla_2(di, dec_weight_ideal_value, i);
			infilled_weights[i] = weight;
		}
	}
	else
	{
		for (unsigned int i = 0; i < texel_count; i ++)
		{
			float weight = bilinear_infill_vla(di, dec_weight_ideal_value, i);
			infilled_weights[i] = weight;
		}
	}

	// Perform a single iteration of refinement
	// Empirically determined step size; larger values don't help but smaller drops image quality
	constexpr float stepsize = 0.25f;
	constexpr float chd_scale = -WEIGHTS_TEXEL_SUM;

	for (unsigned int i = 0; i < weight_count; i ++)
	{
		float weight_val = dec_weight_ideal_value[i];

		// Accumulate error weighting of all the texels using this weight
		// Start with a small value to avoid div-by-zero later
		float error_change0(1e-10f);
		float error_change1(0.0f);

		int weight_texel_count = di.weight_texel_count[i];
		for (unsigned int j = 0; j < weight_texel_count; j++)
		{
			int texel_index = di.weight_texels_tr[j][i];
			float contrib_weight = di.weights_texel_contribs_tr[j][i];

			if (!constant_wes)
			{
				weight_error_scale = ei.weight_error_scale[texel_index];
			}

			float scale = weight_error_scale * contrib_weight;
			float old_weight = infilled_weights[texel_index];
			float ideal_weight = ei.weights[texel_index];

			error_change0 += contrib_weight * scale;
			error_change1 += (old_weight - ideal_weight) * scale;
		}

		float step = (error_change1 * chd_scale) / error_change0;
		step = clamp(-stepsize, stepsize, step);

		dec_weight_ideal_value[i] = weight_val + step;
	}
}

__device__ void compute_ideal_colors_and_weights_4_comp(const image_block& blk, const partition_info& pi, endpoints_and_weights& ei)
{
	const float error_weight = (blk.channel_weight.x + blk.channel_weight.y + blk.channel_weight.z + blk.channel_weight.w) / 4.0f;
	uint8_t partition_count = pi.partition_count;

	partition_metrics pms[BLOCK_MAX_PARTITIONS];
	compute_avgs_and_dirs_4_comp(pi, blk, pms);

	bool is_constant_wes{ true };
	float partition0_len_sq{ 0.0f };

	for (uint8_t i = 0; i < partition_count; i++)
	{
		float4 dir = pms[i].dir;
		if ((dir.x + dir.y + dir.z)< 0.0f)
		{
			dir = -dir;
		}

		float length_dir = length(dir);
		line4 line{ pms[i].avg, length_dir < 1e-10 ? float4(0.5) : normalize(dir) };
		float lowparam{ 1e10f };
		float highparam{ -1e10f };

		unsigned int partition_texel_count = pi.partition_texel_count[i];
		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			float4 point(blk.data_r[tix], blk.data_g[tix], blk.data_b[tix], blk.data_a[tix]);
			float param = dot(point - line.a, line.b);
			ei.weights[tix] = param;

			lowparam = std::min(param, lowparam);
			highparam = std::max(param, highparam);
		}

		if (highparam <= lowparam)
		{
			lowparam = 0.0f;
			highparam = 1e-7f;
		}

		float length = highparam - lowparam;
		float length_squared = length * length;
		float scale = 1.0f / length;

		if (i == 0)
		{
			partition0_len_sq = length_squared;
		}
		else
		{
			is_constant_wes = is_constant_wes && length_squared == partition0_len_sq;
		}

		ei.ep.endpt0[i] = line.a + line.b * lowparam;
		ei.ep.endpt1[i] = line.a + line.b * highparam;

		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			float idx = (ei.weights[tix] - lowparam) * scale;
			idx = clamp(idx, 0.0, 1.0);
			ei.weights[tix] = idx;
			ei.weight_error_scale[tix] = length_squared * error_weight;
		}
	}

	ei.is_constant_weight_error_scale = is_constant_wes;
}

__device__ void compute_ideal_colors_and_weights_1plane(
	const image_block& blk,
	const partition_info& pi,
	endpoints_and_weights& ei
) {
	bool uses_alpha = !(blk.data_min.w == blk.data_max.w);

	if (uses_alpha)
	{
		compute_ideal_colors_and_weights_4_comp(blk, pi, ei);
	}
	else
	{
#if CUDA_DEBUG
		printf("FGAC_ERROR: compute_ideal_colors_and_weights_1plane");
#endif
		//compute_ideal_colors_and_weights_3_comp(blk, pi, ei, 3);
	}
}

// The available quant levels, stored with a minus 1 bias
__constant__ float quant_levels_m1[12]{
	1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, 9.0f, 11.0f, 15.0f, 19.0f, 23.0f, 31.0f
};


__device__ void compute_quantized_weights_for_decimation(
	const block_size_descriptor& bsd,
	const decimation_info& di,
	float low_bound,
	float high_bound,
	const float* dec_weight_ideal_value,
	float* weight_set_out,
	uint8_t* quantized_weight_set,
	quant_method quant_level
) 
{
	int weight_count = di.weight_count;
	const quant_and_transfer_table& qat = bsd.quant_and_xfer_tables[quant_level];

	int steps_m1(get_quant_level(quant_level) - 1);
	float quant_level_m1 = quant_levels_m1[quant_level];

	// Quantize the weight set using both the specified low/high bounds and standard 0..1 bounds
	if (high_bound <= low_bound)
	{
		low_bound = 0.0f;
		high_bound = 1.0f;
	}

	float rscale = high_bound - low_bound;
	float scale = 1.0f / rscale;

	float scaled_low_bound = low_bound * scale;
	rscale *= 1.0f / 64.0f;

	for (int i = 0; i < weight_count; i++)
	{
		// equal to (weight - low_bound) / (high_bound - low_bound)
		float ix = dec_weight_ideal_value[i] * scale - scaled_low_bound;
		ix = clamp(ix, 0.0, 1.0);

		// Look up the two closest indexes and return the one that was closest
		float ix1 = ix * quant_level_m1;

		int weightl = int(ix1);
		int weighth = std::min(weightl + 1, steps_m1);

		int ixli = qat.quant_to_unquant[weightl];
		int ixhi = qat.quant_to_unquant[weighth];

		float ixl = float(ixli);
		float ixh = float(ixhi);

		int weight = 0;
		if (ixl + ixh < 128.0f * ix)
		{
			weight = ixhi;
			ixl = ixh;
		}
		else
		{
			weight = ixli;
			ixl = ixl;
		}

		// Invert the weight-scaling that was done initially
		weight_set_out[i] = ixl * rscale + low_bound;
		quantized_weight_set[i] = weight & 0xFF;
	}
}

__device__ float compute_error_of_weight_set_1plane(
	const endpoints_and_weights& eai,
	const decimation_info& di,
	const float* dec_weight_quant_uvalue)
{
	float error_summa = 0;
	unsigned int texel_count = di.texel_count;
	if (di.max_texel_weight_count > 2)
	{
		for (unsigned int i = 0; i < texel_count; i++)
		{
			// Compute the bilinear interpolation of the decimated weight grid
			float current_values = bilinear_infill_vla(di, dec_weight_quant_uvalue, i);

			float actual_values = eai.weights[i];
			float diff = current_values - actual_values;
			float significance = eai.weight_error_scale[i];
			float error = diff * diff * significance;

			error_summa += error;
		}
	}
	else if (di.max_texel_weight_count > 1)
	{
		for (unsigned int i = 0; i < texel_count; i ++)
		{
			// Compute the bilinear interpolation of the decimated weight grid
			float current_values = bilinear_infill_vla_2(di, dec_weight_quant_uvalue, i);

			// Compute the error between the computed value and the ideal weight
			float actual_values = eai.weights[i];
			float diff = current_values - actual_values;
			float significance = eai.weight_error_scale[i];
			float error = diff * diff * significance;

			error_summa += error;
		}
	}
	else
	{
		for (unsigned int i = 0; i < texel_count; i ++)
		{
			// Load the weight set directly, without interpolation
			float current_values = dec_weight_quant_uvalue[i];

			// Compute the error between the computed value and the ideal weight
			float actual_values = eai.weights[i];
			float diff = current_values - actual_values;
			float significance = eai.weight_error_scale[i];
			float error = diff * diff * significance;

			error_summa += error;
		}
	}
	return error_summa;
}
#endif