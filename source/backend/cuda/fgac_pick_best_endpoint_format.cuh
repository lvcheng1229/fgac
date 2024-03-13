#ifndef _FGAC_PICK_BEST_ENDPOINT_FORMAT_CUH_
#define _FGAC_PICK_BEST_ENDPOINT_FORMAT_CUH_

#include "fgac_internal.cuh"
#include "fgac_averages_and_directions.cuh"

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
		float4 endpt_diff = abs(endpt1 - endpt0);
		vmask4 endpt_can_offset = endpt_diff < vfloat4(0.12f * 65535.0f);
		bool can_offset_encode = (mask(endpt_can_offset) & 0x7) == 0x7;

		// Store out the settings
		eci[i].rgb_scale_error = (samechroma_rgb_error - uncorr_rgb_error) * 0.7f;  // empirical
		eci[i].rgb_luma_error = (rgb_luma_error - uncorr_rgb_error) * 1.5f;        // wild guess
		eci[i].luminance_error = (luminance_rgb_error - uncorr_rgb_error) * 3.0f;   // empirical
		eci[i].alpha_drop_error = alpha_drop_error * 3.0f;
		eci[i].can_offset_encode = can_offset_encode;
		eci[i].can_blue_contract = !blk.is_luminance();
	}
}

__device__ unsigned int compute_ideal_endpoint_formats(
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
	//int partition_count = pi.partition_count;
	
	// Compute the errors that result from various encoding choices (such as using luminance instead
	// of RGB, discarding Alpha, using RGB-scale in place of two separate RGB endpoints and so on)
	encoding_choice_errors eci[BLOCK_MAX_PARTITIONS];
	compute_encoding_choice_errors(blk, pi, ep, eci);
}
#endif