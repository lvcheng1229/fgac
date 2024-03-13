#ifndef _FGAC_DECOMPRESS_SYMBOLIC_CUH_
#define _FGAC_DECOMPRESS_SYMBOLIC_CUH_
#include "fgac_internal.cuh"
#include "fgac_compress_texture.h"

__device__ void unpack_color_endpoints(
	int format,
	const uint8_t* input,
	int4& output0,
	int4& output1
) {

	if (format == FMT_RGB)
	{
		int4 input0q(input[0], input[2], input[4], 0);
		int4 input1q(input[1], input[3], input[5], 0);

		output0 = input0q;
		output1 = input1q;
	}
	else if (format == FMT_RGBA)
	{
		int4 input0q(input[0], input[2], input[4], input[6]);
		int4 input1q(input[1], input[3], input[5], input[7]);
		output0 = input0q;
		output1 = input1q;
	}
	else
	{
#if CUDA_DEBUG

#endif
	}
	output0 = int4(output0.x * 257, output0.y * 257, output0.z * 257, output0.w * 257);
	output1 = int4(output1.x * 257, output1.y * 257, output1.z * 257, output1.w * 257);
}

__device__ void unpack_weights(
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	const decimation_info& di,
	bool is_dual_plane,
	int weights_plane1[BLOCK_MAX_TEXELS],
	int weights_plane2[BLOCK_MAX_TEXELS]
) 
{
	for (unsigned int i = 0; i < bsd.texel_count; i ++)
	{
		int summed_value(8);
		int weight_count(di.texel_weight_count[i]);
		
		for (int j = 0; j < weight_count; j++)
		{
			int texel_weights(di.texel_weights_tr[j][i]);
			int texel_weights_int(di.texel_weight_contribs_int_tr[j][i]);

			summed_value += scb.weights[texel_weights] * texel_weights_int;
		}

		weights_plane1[i] = summed_value >> 4;
	}
}
__device__ float compute_symbolic_block_difference_1plane_1partition(
	const fgac_config& config,
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	const image_block& blk)
{
	// If we detected an error-block, blow up immediately.
	if (scb.block_type == SYM_BTYPE_ERROR)
	{
		return ERROR_CALC_DEFAULT;
	}

	// Get the appropriate block descriptor
	const block_mode& bm = get_block_mode(&bsd,scb.block_mode);
	const decimation_info& di = get_decimation_info(&bsd,bm.decimation_mode);

	int plane1_weights[BLOCK_MAX_TEXELS];
	unpack_weights(bsd, scb, di, false, plane1_weights, nullptr);

	// Decode the color endpoints for this partition
	int4 ep0;
	int4 ep1;
	bool rgb_lns;
	bool a_lns;

	unpack_color_endpoints(
		scb.color_formats[0],
		scb.color_values[0],
		ep0, ep1);
	
	float summa = 0;
	unsigned int texel_count = bsd.texel_count;
	for (unsigned int i = 0; i < texel_count; i ++)
	{
		// Compute EP1 contribution
		int weight1 = plane1_weights[i];
		int ep1_r = ep1.x * weight1;
		int ep1_g = ep1.y * weight1;
		int ep1_b = ep1.z * weight1;
		int ep1_a = ep1.w * weight1;

		// Compute EP0 contribution
		int weight0 = int(64) - weight0;
		int ep0_r = ep0.x * weight0;
		int ep0_g = ep0.y * weight0;
		int ep0_b = ep0.z * weight0;
		int ep0_a = ep0.w * weight0;

		// Combine contributions
		int colori_r = int(ep0_r + ep1_r + int(32)) >> 6;
		int colori_g = int(ep0_g + ep1_g + int(32)) >> 6;
		int colori_b = int(ep0_b + ep1_b + int(32)) >> 6;
		int colori_a = int(ep0_a + ep1_a + int(32)) >> 6;

		// Compute color diff
		float color_r = float(colori_r);
		float color_g = float(colori_g);
		float color_b = float(colori_b);
		float color_a = float(colori_a);

		float color_orig_r = blk.data_r[i];
		float color_orig_g = blk.data_g[i];
		float color_orig_b = blk.data_b[i];
		float color_orig_a = blk.data_a[i];

		float color_error_r = fmin(abs(color_orig_r - color_r), float(1e15f));
		float color_error_g = fmin(abs(color_orig_g - color_g), float(1e15f));
		float color_error_b = fmin(abs(color_orig_b - color_b), float(1e15f));
		float color_error_a = fmin(abs(color_orig_a - color_a), float(1e15f));

		// Compute squared error metric
		color_error_r = color_error_r * color_error_r;
		color_error_g = color_error_g * color_error_g;
		color_error_b = color_error_b * color_error_b;
		color_error_a = color_error_a * color_error_a;

		float metric = color_error_r * blk.channel_weight.x
			+ color_error_g * blk.channel_weight.y
			+ color_error_b * blk.channel_weight.z
			+ color_error_a * blk.channel_weight.w;

		// Mask off bad lanes
		summa += metric;
	}

	return summa;
}
#endif