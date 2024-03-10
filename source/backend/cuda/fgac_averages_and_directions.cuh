#ifndef _FGAC_AVERAGES_AND_DIRECTION_CUH_
#define _FGAC_AVERAGES_AND_DIRECTION_CUH_
#include "fgac_internal.cuh"
#include "common/helper_math.h"

//todo: for (unsigned int i = 0; i < texel_count; i++)
//todo: data_r data_g ... to data_4_comp

__device__ void compute_partition_averages_rgba(
	const partition_info& pi,
	const image_block& blk,
	float4 averages[BLOCK_MAX_PARTITIONS]
) 
{
	unsigned int partition_count = pi.partition_count;
	unsigned int texel_count = blk.texel_count;

	// For 1 partition just use the precomputed mean
	if (partition_count == 1)
	{
		averages[0] = blk.data_mean;
	}
	// For 2 partitions scan results for partition 0, compute partition 1
	else if (partition_count == 2)
	{
		float4 pp_avg_rgba(0);

		for (unsigned int i = 0; i < texel_count; i++)
		{
			uint8_t texel_partition = pi.partition_of_texel[i];
			pp_avg_rgba += (texel_partition == 0) ? make_float4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]) : float4(0);
		}

		float4 block_total = blk.data_mean * static_cast<float>(blk.texel_count);
		
		float4 p0_total = pp_avg_rgba;
		float4 p1_total = block_total - p0_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
	}
	// For 3 partitions scan results for partition 0/1, compute partition 2
	else if (partition_count == 3)
	{
		//patrition 0 and partition 1
		float4 pp_avg_rgba[2] = { 0,0 };
		for (unsigned int i = 0; i < texel_count; i++)
		{
			uint8_t texel_partition = pi.partition_of_texel[i];
			pp_avg_rgba[0] += (texel_partition == 0) ? make_float4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]) : float4(0);
			pp_avg_rgba[1] += (texel_partition == 1) ? make_float4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]) : float4(0);
		}

		float4 block_total = blk.data_mean * static_cast<float>(blk.texel_count);

		float4 p0_total = pp_avg_rgba[0];
		float4 p1_total = pp_avg_rgba[1];
		float4 p2_total = block_total - p0_total - p1_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
		averages[2] = p2_total / static_cast<float>(pi.partition_texel_count[2]);
	}
	else
	{
		// For 4 partitions scan results for partition 0/1/2, compute partition 3
		//patrition 0 and partition 1
		float4 pp_avg_rgba[3] = { 0,0 ,0 };
		float4 pp_avg_rgba[2] = { 0,0 };
		for (unsigned int i = 0; i < texel_count; i++)
		{
			uint8_t texel_partition = pi.partition_of_texel[i];
			pp_avg_rgba[0] += (texel_partition == 0) ? make_float4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]) : float4(0);
			pp_avg_rgba[1] += (texel_partition == 1) ? make_float4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]) : float4(0);
			pp_avg_rgba[2] += (texel_partition == 2) ? make_float4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]) : float4(0);
		}

		float4 block_total = blk.data_mean * static_cast<float>(blk.texel_count);

		float4 p0_total = pp_avg_rgba[0];
		float4 p1_total = pp_avg_rgba[1];
		float4 p2_total = pp_avg_rgba[2];
		float4 p3_total = block_total - p0_total - p1_total - p2_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
		averages[2] = p2_total / static_cast<float>(pi.partition_texel_count[2]);
		averages[3] = p3_total / static_cast<float>(pi.partition_texel_count[3]);
	}
}

__device__ void compute_avgs_and_dirs_4_comp(
	const partition_info& pi,
	const image_block& blk,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
) 
{
	int partition_count = pi.partition_count;
	float4 partition_averages[BLOCK_MAX_PARTITIONS];
	compute_partition_averages_rgba(pi, blk, partition_averages);

	for (int partition = 0; partition < partition_count; partition++)
	{
		const uint8_t* texel_indexes = pi.texels_of_partition[partition];
		unsigned int texel_count = pi.partition_texel_count[partition];

		float4 average = partition_averages[partition];
		pm[partition].avg = average;

		float4 sum_xp(0.0);
		float4 sum_yp(0.0);
		float4 sum_zp(0.0);
		float4 sum_wp(0.0);

		for (unsigned int i = 0; i < texel_count; i++)
		{
			uint8_t iwt = texel_indexes[i];
			float4 texel_datum = make_float4(blk.data_r[iwt], blk.data_g[iwt], blk.data_b[iwt], blk.data_a[iwt]);
			texel_datum = texel_datum - average;

			sum_xp += (texel_datum.x > 0) ? texel_datum : float4(0);
			sum_yp += (texel_datum.y > 0) ? texel_datum : float4(0);
			sum_zp += (texel_datum.z > 0) ? texel_datum : float4(0);
			sum_wp += (texel_datum.w > 0) ? texel_datum : float4(0);
		}

		float prod_xp = dot(sum_xp, sum_xp);
		float prod_yp = dot(sum_yp, sum_yp);
		float prod_zp = dot(sum_zp, sum_zp);
		float prod_wp = dot(sum_wp, sum_wp);

		float4 best_vector = sum_xp;
		float best_sum = prod_xp;

		if (prod_yp > best_sum)
		{
			best_vector = sum_yp;
			best_sum = prod_yp;
		}

		if (prod_zp > best_sum)
		{
			best_vector = sum_zp;
			best_sum = prod_zp;
		}

		if (prod_wp > best_sum)
		{
			best_vector = sum_wp;
			best_sum = prod_wp;
		}

		pm[partition].dir = best_vector;
	}
}
#endif