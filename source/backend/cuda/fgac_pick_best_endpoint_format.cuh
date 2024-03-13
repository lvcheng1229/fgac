#ifndef _FGAC_PICK_BEST_ENDPOINT_FORMAT_CUH_
#define _FGAC_PICK_BEST_ENDPOINT_FORMAT_CUH_

#include "fgac_internal.cuh"
#include "fgac_averages_and_directions.cuh"

__device__ void compute_error_squared_rgb_single_partition(
	const partition_info& pi,
	int partition_index,
	const image_block& blk,
	const processed_line3& uncor_pline,
	float& uncor_err,
	const processed_line3& samec_pline,
	float& samec_err,
	const processed_line3& rgbl_pline,
	float& rgbl_err,
	const processed_line3& l_pline,
	float& l_err,
	float& a_drop_err
) 
{
	float4 ews = blk.channel_weight;

	unsigned int texel_count = pi.partition_texel_count[partition_index];
	const uint8_t* texel_indexes = pi.texels_of_partition[partition_index];

	float a_drop_err = 0;
	float default_a(0xFFFF);

	float uncor_bs0(uncor_pline.bs.x);
	float uncor_bs1(uncor_pline.bs.y);
	float uncor_bs2(uncor_pline.bs.z);

	float uncor_amod0(uncor_pline.amod.x);
	float uncor_amod1(uncor_pline.amod.y);
	float uncor_amod2(uncor_pline.amod.z);

	float samec_bs0(samec_pline.bs.x);
	float samec_bs1(samec_pline.bs.y);
	float samec_bs2(samec_pline.bs.z);

	float rgbl_bs0(rgbl_pline.bs.x);
	float rgbl_bs1(rgbl_pline.bs.y);
	float rgbl_bs2(rgbl_pline.bs.z);

	float rgbl_amod0(rgbl_pline.amod.x);
	float rgbl_amod1(rgbl_pline.amod.y);
	float rgbl_amod2(rgbl_pline.amod.z);

	float l_bs0(l_pline.bs.x);
	float l_bs1(l_pline.bs.y);
	float l_bs2(l_pline.bs.z);

	for (unsigned int i = 0; i < texel_count; i ++)
	{
		int tix(texel_indexes[i]);

		// Compute the error that arises from just ditching alpha
		float data_a = blk.data_a[tix];
		float alpha_diff = data_a - default_a;
		alpha_diff = alpha_diff * alpha_diff;

		a_drop_err += alpha_diff;

		float data_r = blk.data_r[tix];
		float data_g = blk.data_g[tix];
		float data_b = blk.data_b[tix];

		// Compute uncorrelated error
		float param = data_r * uncor_bs0 + data_g * uncor_bs1 + data_b * uncor_bs2;

		float dist0 = (uncor_amod0 + param * uncor_bs0) - data_r;
		float dist1 = (uncor_amod1 + param * uncor_bs1) - data_g;
		float dist2 = (uncor_amod2 + param * uncor_bs2) - data_b;

		float error = dist0 * dist0 * ews.x
			+ dist1 * dist1 * ews.y
			+ dist2 * dist2 * ews.z;

		uncor_err += error;

		// Compute same chroma error - no "amod", its always zero
		param = data_r * samec_bs0
			+ data_g * samec_bs1
			+ data_b * samec_bs2;

		dist0 = (param * samec_bs0) - data_r;
		dist1 = (param * samec_bs1) - data_g;
		dist2 = (param * samec_bs2) - data_b;

		error = dist0 * dist0 * ews.x
			+ dist1 * dist1 * ews.y
			+ dist2 * dist2 * ews.z;

		samec_err += error;

		// Compute rgbl error
		param = data_r * rgbl_bs0
			+ data_g * rgbl_bs1
			+ data_b * rgbl_bs2;

		dist0 = (rgbl_amod0 + param * rgbl_bs0) - data_r;
		dist1 = (rgbl_amod1 + param * rgbl_bs1) - data_g;
		dist2 = (rgbl_amod2 + param * rgbl_bs2) - data_b;

		error = dist0 * dist0 * ews.x
			+ dist1 * dist1 * ews.y
			+ dist2 * dist2 * ews.z;

		rgbl_err += error;

		// Compute luma error - no "amod", its always zero
		param = data_r * l_bs0
			+ data_g * l_bs1
			+ data_b * l_bs2;

		dist0 = (param * l_bs0) - data_r;
		dist1 = (param * l_bs1) - data_g;
		dist2 = (param * l_bs2) - data_b;

		error = dist0 * dist0 * ews.x
			+ dist1 * dist1 * ews.y
			+ dist2 * dist2 * ews.z;

		l_err += error;
	}

	a_drop_err *= ews.w;
}

__device__ void compute_encoding_choice_errors(
	const image_block& blk,
	const partition_info& pi,
	const endpoints& ep,
	encoding_choice_errors eci[BLOCK_MAX_PARTITIONS])
{
	int partition_count = pi.partition_count;

	partition_metrics pms[BLOCK_MAX_PARTITIONS];

	compute_avgs_and_dirs_3_comp_rgb(pi, blk, pms);

	for (int i = 0; i < partition_count; i++)
	{
		partition_metrics& pm = pms[i];

		line3 uncor_rgb_lines;
		line3 samec_rgb_lines;  // for LDR-RGB-scale
		line3 rgb_luma_lines;   // for HDR-RGB-scale

		processed_line3 uncor_rgb_plines;
		processed_line3 samec_rgb_plines;
		processed_line3 rgb_luma_plines;
		processed_line3 luminance_plines;

		float uncorr_rgb_error;
		float samechroma_rgb_error;
		float rgb_luma_error;
		float luminance_rgb_error;
		float alpha_drop_error;

		uncor_rgb_lines.a = pm.avg;
		uncor_rgb_lines.b = normalize_safe(pm.dir);

		samec_rgb_lines.a = float4(0);
		samec_rgb_lines.b = normalize_safe(pm.avg);

		float val = 0.577350258827209473f;

		rgb_luma_lines.a = pm.avg;
		rgb_luma_lines.b = float4(val, val, val, 0.0f);

		uncor_rgb_plines.amod = uncor_rgb_lines.a - uncor_rgb_lines.b * dot(uncor_rgb_lines.a, uncor_rgb_lines.b);
		uncor_rgb_plines.bs = uncor_rgb_lines.b;

		// Same chroma always goes though zero, so this is simpler than the others
		samec_rgb_plines.amod = float4(0);
		samec_rgb_plines.bs = samec_rgb_lines.b;

		rgb_luma_plines.amod = rgb_luma_lines.a - rgb_luma_lines.b * dot(rgb_luma_lines.a, rgb_luma_lines.b);
		rgb_luma_plines.bs = rgb_luma_lines.b;

		// Luminance always goes though zero, so this is simpler than the others
		luminance_plines.amod = float4(0);
		luminance_plines.bs = float4(val, val, val, 0.0f);

		compute_error_squared_rgb_single_partition(
			pi, i, blk,
			uncor_rgb_plines, uncorr_rgb_error,
			samec_rgb_plines, samechroma_rgb_error,
			rgb_luma_plines, rgb_luma_error,
			luminance_plines, luminance_rgb_error,
			alpha_drop_error);

		// Determine if we can offset encode RGB lanes
		float4 endpt0 = ep.endpt0[i];
		float4 endpt1 = ep.endpt1[i];
		float4 endpt_diff = fabs(endpt1 - endpt0);
		bool can_offset_encode = endpt_diff.x < float(0.12f * 65535.0f) && endpt_diff.y < float(0.12f * 65535.0f) && endpt_diff.z < float(0.12f * 65535.0f) && endpt_diff.w < float(0.12f * 65535.0f);

		// Store out the settings
		eci[i].rgb_scale_error = (samechroma_rgb_error - uncorr_rgb_error) * 0.7f;  // empirical
		eci[i].rgb_luma_error = (rgb_luma_error - uncorr_rgb_error) * 1.5f;        // wild guess
		eci[i].luminance_error = (luminance_rgb_error - uncorr_rgb_error) * 3.0f;   // empirical
		eci[i].alpha_drop_error = alpha_drop_error * 3.0f;
		eci[i].can_offset_encode = can_offset_encode;
		//eci[i].can_blue_contract = !blk.is_luminance();
	}
}

__constant__ float baseline_quant_error[21 - QUANT_6]{
	(65536.0f * 65536.0f / 18.0f) / (5 * 5),
	(65536.0f * 65536.0f / 18.0f) / (7 * 7),
	(65536.0f * 65536.0f / 18.0f) / (9 * 9),
	(65536.0f * 65536.0f / 18.0f) / (11 * 11),
	(65536.0f * 65536.0f / 18.0f) / (15 * 15),
	(65536.0f * 65536.0f / 18.0f) / (19 * 19),
	(65536.0f * 65536.0f / 18.0f) / (23 * 23),
	(65536.0f * 65536.0f / 18.0f) / (31 * 31),
	(65536.0f * 65536.0f / 18.0f) / (39 * 39),
	(65536.0f * 65536.0f / 18.0f) / (47 * 47),
	(65536.0f * 65536.0f / 18.0f) / (63 * 63),
	(65536.0f * 65536.0f / 18.0f) / (79 * 79),
	(65536.0f * 65536.0f / 18.0f) / (95 * 95),
	(65536.0f * 65536.0f / 18.0f) / (127 * 127),
	(65536.0f * 65536.0f / 18.0f) / (159 * 159),
	(65536.0f * 65536.0f / 18.0f) / (191 * 191),
	(65536.0f * 65536.0f / 18.0f) / (255 * 255)
};

__device__ float one_partition_find_best_combination_for_bitcount(
	const block_size_descriptor& bsd,
	const float best_combined_error[21][4],
	const uint8_t best_combined_format[21][4],
	int bits_available,
	uint8_t& best_quant_level,
	uint8_t& best_format
) {
	int best_integer_count = 0;
	float best_integer_count_error = ERROR_CALC_DEFAULT;

	for (int integer_count = 1; integer_count <= 4; integer_count++)
	{
		// Compute the quantization level for a given number of integers and a given number of bits
		int quant_level = bsd.quant_mode_table[integer_count][bits_available];

		// Don't have enough bits to represent a given endpoint format at all!
		if (quant_level < QUANT_6)
		{
			continue;
		}

		float integer_count_error = best_combined_error[quant_level][integer_count - 1];
		if (integer_count_error < best_integer_count_error)
		{
			best_integer_count_error = integer_count_error;
			best_integer_count = integer_count - 1;
		}
	}

	int ql = bsd.quant_mode_table[best_integer_count + 1][bits_available];

	best_quant_level = static_cast<uint8_t>(ql);
	best_format = FMT_LUMINANCE;

	if (ql >= QUANT_6)
	{
		best_format = best_combined_format[ql][best_integer_count];
	}

	return best_integer_count_error;
}

__device__ void compute_color_error_for_every_integer_count_and_quant_level(
	int partition_index,
	const partition_info& pi,
	const encoding_choice_errors& eci,
	const endpoints& ep,
	float4 error_weight,
	float best_error[21][4],
	uint8_t format_of_choice[21][4]
) 
{
	int partition_size = pi.partition_texel_count[partition_index];

	float4 ep0 = ep.endpt0[partition_index];
	float4 ep1 = ep.endpt1[partition_index];

	float ep1_min = std::min(std::min(ep1.x, ep1.y), std::min(ep1.z, ep1.w));
	ep1_min = std::max(ep1_min, 0.0f);

	float error_weight_rgbsum = error_weight.x + error_weight.y + error_weight.z + error_weight.w;

	// It is possible to get endpoint colors significantly outside [0,upper-limit] even if the
	// input data are safely contained in [0,upper-limit]; we need to add an error term for this
	float4 offset(65535.0f, 65535.0f, 65535.0f, 65535.0f);
	float4 ep0_range_error_high = fmaxf(ep0 - offset, float4(0.0f));
	float4 ep1_range_error_high = fmaxf(ep1 - offset, float4(0.0f));

	float4 ep0_range_error_low = fminf(ep0, float4(0.0f));
	float4 ep1_range_error_low = fminf(ep1, float4(0.0f));

	float4 sum_range_error =
		(ep0_range_error_low * ep0_range_error_low) +
		(ep1_range_error_low * ep1_range_error_low) +
		(ep0_range_error_high * ep0_range_error_high) +
		(ep1_range_error_high * ep1_range_error_high);

	float3 sum_range_error_rgb(sum_range_error.x, sum_range_error.y, sum_range_error.z);
	float3 error_weight_rgb(error_weight.x, error_weight.y, error_weight.z);
	float rgb_range_error = dot(sum_range_error_rgb, error_weight_rgb) * 0.5f * static_cast<float>(partition_size);
	float alpha_range_error = sum_range_error.w * error_weight.w * 0.5f * static_cast<float>(partition_size);

	{
		for (int i = QUANT_2; i < QUANT_6; i++)
		{
			best_error[i][3] = ERROR_CALC_DEFAULT;
			best_error[i][2] = ERROR_CALC_DEFAULT;
			best_error[i][1] = ERROR_CALC_DEFAULT;
			best_error[i][0] = ERROR_CALC_DEFAULT;

			format_of_choice[i][3] = FMT_RGBA;
			format_of_choice[i][2] = FMT_RGB;
			format_of_choice[i][1] = FMT_RGB_SCALE;
			format_of_choice[i][0] = FMT_LUMINANCE;
		}

		float base_quant_error_rgb = error_weight_rgbsum * static_cast<float>(partition_size);
		float base_quant_error_a = error_weight.w * static_cast<float>(partition_size);
		float base_quant_error_rgba = base_quant_error_rgb + base_quant_error_a;

		float error_scale_oe_rgba = eci.can_offset_encode ? 0.5f : 1.0f;

		float error_scale_oe_rgb = eci.can_offset_encode ? 0.25f : 1.0f;

		// Pick among the available LDR endpoint modes
		for (int i = QUANT_6; i <= QUANT_256; i++)
		{
			// Offset encoding not possible at higher quant levels
			if (i >= QUANT_192)
			{
				error_scale_oe_rgba = 1.0f;
				error_scale_oe_rgb = 1.0f;
			}

			float base_quant_error = baseline_quant_error[i - QUANT_6];
			float quant_error_rgb = base_quant_error_rgb * base_quant_error;
			float quant_error_rgba = base_quant_error_rgba * base_quant_error;

			// 8 integers can encode as RGBA+RGBA
			float full_ldr_rgba_error = quant_error_rgba
				* error_scale_oe_rgba
				+ rgb_range_error
				+ alpha_range_error;

			best_error[i][3] = full_ldr_rgba_error;
			format_of_choice[i][3] = FMT_RGBA;

			// 6 integers can encode as RGB+RGB or RGBS+AA
			float full_ldr_rgb_error = quant_error_rgb
				* error_scale_oe_rgb
				+ rgb_range_error
				+ eci.alpha_drop_error;

			float rgbs_alpha_error = quant_error_rgba
				+ eci.rgb_scale_error
				+ rgb_range_error
				+ alpha_range_error;

			if (rgbs_alpha_error < full_ldr_rgb_error)
			{
				best_error[i][2] = rgbs_alpha_error;
				format_of_choice[i][2] = FMT_RGB_SCALE_ALPHA;
			}
			else
			{
				best_error[i][2] = full_ldr_rgb_error;
				format_of_choice[i][2] = FMT_RGB;
			}

			// 4 integers can encode as RGBS or LA+LA
			float ldr_rgbs_error = quant_error_rgb
				+ rgb_range_error
				+ eci.alpha_drop_error
				+ eci.rgb_scale_error;

			float lum_alpha_error = quant_error_rgba
				+ rgb_range_error
				+ alpha_range_error
				+ eci.luminance_error;

			if (ldr_rgbs_error < lum_alpha_error)
			{
				best_error[i][1] = ldr_rgbs_error;
				format_of_choice[i][1] = FMT_RGB_SCALE;
			}
			else
			{
				best_error[i][1] = lum_alpha_error;
				format_of_choice[i][1] = FMT_LUMINANCE_ALPHA;
			}

			// 2 integers can encode as L+L
			float luminance_error = quant_error_rgb
				+ rgb_range_error
				+ eci.alpha_drop_error
				+ eci.luminance_error;

			best_error[i][0] = luminance_error;
			format_of_choice[i][0] = FMT_LUMINANCE;
		}
	}

}
__device__ unsigned int compute_ideal_endpoint_formats(
	const block_size_descriptor& bsd,
	const partition_info& pi,
	const image_block& blk,
	const endpoints& ep,
	// bitcounts and errors computed for the various quantization methods
	const int8_t* qwt_bitcounts,
	const float* qwt_errors,
	unsigned int tune_candidate_limit,
	unsigned int start_block_mode,
	unsigned int end_block_mode,
	// output data
	uint8_t partition_format_specifiers[TUNE_MAX_TRIAL_CANDIDATES][BLOCK_MAX_PARTITIONS],
	int block_mode[TUNE_MAX_TRIAL_CANDIDATES],
	quant_method quant_level[TUNE_MAX_TRIAL_CANDIDATES],
	quant_method quant_level_mod[TUNE_MAX_TRIAL_CANDIDATES],
	compression_working_buffers& tmpbuf
) 
{
	// Compute the errors that result from various encoding choices (such as using luminance instead
	// of RGB, discarding Alpha, using RGB-scale in place of two separate RGB endpoints and so on)
	encoding_choice_errors eci[BLOCK_MAX_PARTITIONS];
	compute_encoding_choice_errors(blk, pi, ep, eci);

	float best_error[BLOCK_MAX_PARTITIONS][21][4];
	uint8_t format_of_choice[BLOCK_MAX_PARTITIONS][21][4];
	for (int index = 0; index < 1; index++)
	{
		compute_color_error_for_every_integer_count_and_quant_level(index,
			pi, eci[index], ep, blk.channel_weight, best_error[index],
			format_of_choice[index]);
	}

	float* errors_of_best_combination = tmpbuf.errors_of_best_combination;
	uint8_t* best_quant_levels = tmpbuf.best_quant_levels;
	uint8_t* best_quant_levels_mod = tmpbuf.best_quant_levels_mod;
	uint8_t(&best_ep_formats)[WEIGHTS_MAX_BLOCK_MODES][BLOCK_MAX_PARTITIONS] = tmpbuf.best_ep_formats;

	
	float clear_error(ERROR_CALC_DEFAULT);
	int clear_quant(0);

	// Ensure that the first iteration understep contains data that will never be picked
	errors_of_best_combination[start_block_mode] = clear_error;
	best_quant_levels[start_block_mode] = clear_quant;
	best_quant_levels_mod[start_block_mode] = clear_quant;
	
	// Ensure that last iteration overstep contains data that will never be picked
	errors_of_best_combination[end_block_mode - 1] = clear_error;
	best_quant_levels[end_block_mode - 1] = clear_quant;
	best_quant_levels_mod[end_block_mode - 1] = clear_quant;

	// Track a scalar best to avoid expensive search at least once ...
	float error_of_best_combination = ERROR_CALC_DEFAULT;
	int index_of_best_combination = -1;

	// The block contains 1 partition
	{
		for (unsigned int i = start_block_mode; i < end_block_mode; i++)
		{
			if (qwt_errors[i] >= ERROR_CALC_DEFAULT)
			{
				errors_of_best_combination[i] = ERROR_CALC_DEFAULT;
				continue;
			}

			float error_of_best = one_partition_find_best_combination_for_bitcount(
				bsd, best_error[0], format_of_choice[0], qwt_bitcounts[i],
				best_quant_levels[i], best_ep_formats[i][0]);

			float total_error = error_of_best + qwt_errors[i];
			errors_of_best_combination[i] = total_error;
			best_quant_levels_mod[i] = best_quant_levels[i];

			if (total_error < error_of_best_combination)
			{
				error_of_best_combination = total_error;
				index_of_best_combination = i;
			}
		}
	}

	int best_error_weights[TUNE_MAX_TRIAL_CANDIDATES];

	// Fast path the first result and avoid the list search for trial 0
	best_error_weights[0] = index_of_best_combination;
	if (index_of_best_combination >= 0)
	{
		errors_of_best_combination[index_of_best_combination] = ERROR_CALC_DEFAULT;
	}

	// Search the remaining results and pick the best candidate modes for trial 1+
	for (unsigned int i = 1; i < tune_candidate_limit; i++)
	{
		int best_error_index(-1);
		float best_ep_error(ERROR_CALC_DEFAULT);

		for (unsigned int j = start_block_mode; j < end_block_mode; j ++)
		{
			float err = errors_of_best_combination[j];
			best_ep_error = err < best_ep_error ? err : best_ep_error; 
			best_error_index = err < best_ep_error ? j :best_error_index;
		}

		best_error_weights[i] = best_error_index;

		// Max the error for this candidate so we don't pick it again
		if (best_error_index >= 0)
		{
			errors_of_best_combination[best_error_index] = ERROR_CALC_DEFAULT;
		}
		// Early-out if no more candidates are valid
		else
		{
			break;
		}
	}

	for (unsigned int i = 0; i < tune_candidate_limit; i++)
	{
		if (best_error_weights[i] < 0)
		{
			return i;
		}

		block_mode[i] = best_error_weights[i];

		quant_level[i] = static_cast<quant_method>(best_quant_levels[best_error_weights[i]]);
		quant_level_mod[i] = static_cast<quant_method>(best_quant_levels_mod[best_error_weights[i]]);


		int partition_count = 1;
		for (int j = 0; j < partition_count; j++)
		{
			partition_format_specifiers[i][j] = best_ep_formats[best_error_weights[i]][j];
		}
	}

	return tune_candidate_limit;

}
#endif