#ifndef _FGAC_IDEAL_ENDPOINTS_AND_WEIGHTS_CUH_
#define _FGAC_IDEAL_ENDPOINTS_AND_WEIGHTS_CUH_
#include "fgac_decvice_common.cuh"
#include "fgac_averages_and_directions.cuh"

__device__ void compute_ideal_colors_and_weights_4_comp(const image_block& blk, endpoints_and_weights& ei)
{
	float4 dir;
	compute_avgs_and_dirs_4_comp(blk, dir);

	if ((dir.x + dir.y + dir.z) < 0.0f)
	{
		dir = -dir;
	}

	float length_dir = length(dir);
	line4 line{ blk.data_mean, length_dir < 1e-10 ? float4(0.5) : normalize(dir) };
	float lowparam{ 1e10f };
	float highparam{ -1e10f };

	for (unsigned int j = 0; j < blk.texel_count; j++)
	{
		float4 point(blk.data_r[j], blk.data_g[j], blk.data_b[j], blk.data_a[j]);
		float param = dot(point - line.a, line.b);

		ei.weights[j] = param;

		lowparam = fmin(param, lowparam);
		highparam = fmax(param, highparam);
	}

	if (highparam <= lowparam)
	{
		lowparam = 0.0f;
		highparam = 1e-7f;
	}

	ei.ep.endpt0 = line.a + line.b * lowparam;
	ei.ep.endpt1 = line.a + line.b * highparam;

	float length = highparam - lowparam;
	float length_squared = length * length;
	float scale = 1.0f / length;

	for (unsigned int j = 0; j < blk.texel_count; j++)
	{
		float idx = (ei.weights[j] - lowparam) * scale;
		idx = clamp(idx, 0.0, 1.0);
		ei.weights[j] = idx;
	}
}

__device__ void compute_ideal_colors_and_weights_1plane(
	const image_block& blk,
	endpoints_and_weights& ei
) {
	compute_ideal_colors_and_weights_4_comp(blk, ei);
}

__device__ float compute_error_of_weight_set_1plane(
	const endpoints_and_weights& eai,
	const block_size_descriptor& bsd,
	const float* weight_quant_uvalue)
{
	float error_summa = 0;
	for (unsigned int i = 0; i < bsd.texel_count; i++)
	{
		// Load the weight set directly, without interpolation
		float current_values = weight_quant_uvalue[i];

		// Compute the error between the computed value and the ideal weight
		float actual_values = eai.weights[i];
		float diff = current_values - actual_values;
		float error = diff * diff;

		error_summa += error;
	}
	return error_summa;
}

__device__ void compute_quantized_weights_for_decimation(
	const block_size_descriptor& bsd,
	uint8_t* quantized_weight_set,
	quant_method quant_level
)
{
	if (quant_level <= QUANT_16)
	{

	}
	else
	{

	}
}
#endif