#ifndef _FGAC_COLOR_QUANTIZE_CUH_
#define _FGAC_COLOR_QUANTIZE_CUH_

#include "fgac_internal.cuh"
#include "common/helper_math.h"

__device__ int4 quant_color3(
	const block_size_descriptor& bsd,
	quant_method quant_level,
	int4 value,
	float4 valuef
) 
{
	int4 index = value * 2;

	// Compute the residual to determine if we should round down or up ties.
	// Test should be residual >= 0, but empirical testing shows small bias helps.
	float4 residual = valuef - float4(value.x, value.y, value.z, value.w);

	index = int4(
		residual.x >= 0.1f ? index.x + 1 : index.x,
		residual.y >= 0.1f ? index.y + 1 : index.y,
		residual.z >= 0.1f ? index.z + 1 : index.z,
		residual.w >= 0.1f ? index.w + 1 : index.w
	);

	return int4(
		bsd.color_unquant_to_uquant_tables[quant_level - QUANT_6][index.x],
		bsd.color_unquant_to_uquant_tables[quant_level - QUANT_6][index.y],
		bsd.color_unquant_to_uquant_tables[quant_level - QUANT_6][index.z],
		0);
}

__device__ void quantize_rgb(
	const block_size_descriptor& bsd,
	float4 color0,
	float4 color1,
	int4& color0_out,
	int4& color1_out,
	quant_method quant_level
) {
	int4 color0i, color1i;
	float4 nudge(0.2f);

	do
	{
		float4 round_color = color0 + float4(0.5);
		int4 color0q = max(int4(round_color.x, round_color.y, round_color.z, round_color.w), int4(0));
		color0i = quant_color3(bsd,quant_level, color0q, color0);
		color0 = color0 - nudge;

		int4 color1q = min(int4(round_color.x, round_color.y, round_color.z, round_color.w), int4(255));
		color1i = quant_color3(bsd, quant_level, color1q, color1);
		color1 = color1 + nudge;
	} while ((color0i.x + color0i.y+ color0i.z) > (color1i.x + color1i.y + color1i.z));

	color0_out = color0i;
	color1_out = color1i;
}

__device__ uint8_t quant_color(
	const block_size_descriptor& bsd,
	quant_method quant_level,
	int value,
	float valuef
) {
	int index = value * 2;

	// Compute the residual to determine if we should round down or up ties.
	// Test should be residual >= 0, but empirical testing shows small bias helps.
	float residual = valuef - static_cast<float>(value);
	if (residual >= -0.1f)
	{
		index++;
	}

	return bsd.color_unquant_to_uquant_tables[quant_level - QUANT_6][index];
}

__device__ void quantize_rgba(
	const block_size_descriptor& bsd,
	float4 color0,
	float4 color1,
	int4& color0_out,
	int4& color1_out,
	quant_method quant_level
) {
	quantize_rgb(bsd, color0, color1, color0_out, color1_out, quant_level);

	float a0 = color0.w;
	float a1 = color1.w;

	color0_out.w = quant_color(bsd,quant_level, int(a0+0.5f), a0);
	color1_out.w = quant_color(bsd,quant_level, int(a1+0.5f), a1);
}

__device__ float get_rgba_encoding_error(
	float4 uquant0,
	float4 uquant1,
	int4 quant0,
	int4 quant1
) {
	float4 error0 = uquant0 - float4(quant0.x, quant0.y, quant0.z, quant0.w);
	float4 error1 = uquant1 - float4(quant1.x, quant1.y, quant1.z, quant1.w);
	float4 diff_error = (error0 * error0 + error1 * error1);
	return diff_error.x + diff_error.y + diff_error.z + diff_error.w;
}

__device__ uint8_t pack_color_endpoints(
	const block_size_descriptor& bsd,
	float4 color0,
	float4 color1,
	float4 rgbs_color,
	float4 rgbo_color,
	int format,
	uint8_t* output,
	quant_method quant_level)
{
	// Clamp colors to a valid LDR range
	// Note that HDR has a lower max, handled in the conversion functions
	color0 = clamp(float4(0.0f), float4(65535.0f), color0);
	color1 = clamp(float4(0.0f), float4(65535.0f), color1);

	// Pre-scale the LDR value we need to the 0-255 quantizable range
	float4 color0_ldr = color0 * (1.0f / 257.0f);
	float4 color1_ldr = color1 * (1.0f / 257.0f);

	uint8_t retval = 0;
	float best_error = ERROR_CALC_DEFAULT;
	int4 color0_out, color1_out;
	int4 color0_out2, color1_out2;

	if (format == FMT_RGB)
	{
		quantize_rgb(bsd, color0_ldr, color1_ldr, color0_out2, color1_out2, quant_level);

		int4 color0_unpack = color0_out2;
		int4 color1_unpack = color1_out2;

		float error = get_rgba_encoding_error(color0_ldr, color1_ldr, color0_unpack, color1_unpack);
		if (error < best_error)
		{
			retval = FMT_RGB;
			color0_out = color0_out2;
			color1_out = color1_out2;
		}
	}
	else if (format == FMT_RGBA)
	{
		quantize_rgba(bsd, color0_ldr, color1_ldr, color0_out2, color1_out2, quant_level);

		int4 color0_unpack = color0_out2;
		int4 color1_unpack = color1_out2;

		float error = get_rgba_encoding_error(color0_ldr, color1_ldr, color0_unpack, color1_unpack);
		if (error < best_error)
		{
			retval = FMT_RGBA;
			color0_out = color0_out2;
			color1_out = color1_out2;
		}

		output[6] = static_cast<uint8_t>(color0_out.w);
		output[7] = static_cast<uint8_t>(color1_out.w);
	}
	else
	{
#if CUDA_DEBUG
		printf("error:pack_color_endpoints!");
#endif
	}
	

	// TODO: Can we vectorize this?
	output[0] = static_cast<uint8_t>(color0_out.x);
	output[1] = static_cast<uint8_t>(color1_out.x);
	output[2] = static_cast<uint8_t>(color0_out.y);
	output[3] = static_cast<uint8_t>(color1_out.y);
	output[4] = static_cast<uint8_t>(color0_out.z);
	output[5] = static_cast<uint8_t>(color1_out.z);

}

#endif