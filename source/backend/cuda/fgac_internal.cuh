#ifndef _FGAC_INTERNAL_CUH_
#define _FGAC_INTERNAL_CUH_

#define COMPRESS_ONLY 1
#define CUDA_DEBUG 1

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define __CUDACC__ 1 //todo:
#include <device_functions.h>

#include "fgac_compress_texture.h"

struct line4
{
	float4 a;
	float4 b;
};

struct image_block
{
	float data_r[BLOCK_MAX_TEXELS];
	float data_g[BLOCK_MAX_TEXELS];
	float data_b[BLOCK_MAX_TEXELS];
	float data_a[BLOCK_MAX_TEXELS];

	uint8_t texel_count;

	float4 data_min;
	float4 data_max;
	float4 data_mean;

	float4 channel_weight;
};

//the unpacked content of a single physical compressed block
struct symbolic_compressed_block
{
	uint8_t block_type;/** @brief The block type, one of the @c SYM_BTYPE_* constants. */
	uint8_t partition_count;/** @brief The number of partitions; valid for @c NONCONST blocks. */
	uint8_t color_formats_matched;/** @brief Non-zero if the color formats matched; valid for @c NONCONST blocks. */
	int8_t plane2_component;/** @brief The plane 2 color component, or -1 if single plane; valid for @c NONCONST blocks. */
	uint16_t block_mode;/** @brief The block mode; valid for @c NONCONST blocks. */
	uint16_t partition_index;/** @brief The partition index; valid for @c NONCONST blocks if 2 or more partitions. */
	uint8_t color_formats[BLOCK_MAX_PARTITIONS];/** @brief The endpoint color formats for each partition; valid for @c NONCONST blocks. */
	quant_method quant_mode;/** @brief The endpoint color quant mode; valid for @c NONCONST blocks. */
	float errorval;/** @brief The error of the current encoding; valid for @c NONCONST blocks. */

	// We can't have both of these at the same time
	union {
		/** @brief The constant color; valid for @c CONST blocks. */
		int constant_color[4];

		/** @brief The quantized endpoint color pairs; valid for @c NONCONST blocks. */
		uint8_t color_values[BLOCK_MAX_PARTITIONS][8];
	};

	/** @brief The quantized and decimated weights.
	 *
	 * Weights are stored in the 0-64 unpacked range allowing them to be used
	 * directly in encoding passes without per-use unpacking. Packing happens
	 * when converting to/from the physical bitstream encoding.
	 *
	 * If dual plane, the second plane starts at @c weights[WEIGHTS_PLANE2_OFFSET].
	 */
	uint8_t weights[BLOCK_MAX_WEIGHTS];
};

struct compression_working_buffers
{
	endpoints_and_weights ei1;/** @brief Ideal endpoints and weights for plane 1. */
	endpoints_and_weights ei2;/** @brief Ideal endpoints and weights for plane 2. */

	float dec_weights_ideal[WEIGHTS_MAX_DECIMATION_MODES * BLOCK_MAX_WEIGHTS];//Decimated ideal weight values in the ~0-1 range.
	uint8_t dec_weights_uquant[WEIGHTS_MAX_BLOCK_MODES * BLOCK_MAX_WEIGHTS];//Decimated quantized weight values in the unquantized 0-64 range.

	
	float weight_low_value1[WEIGHTS_MAX_BLOCK_MODES];/** @brief The low weight value in plane 1 for each block mode. */
	float weight_high_value1[WEIGHTS_MAX_BLOCK_MODES];/** @brief The high weight value in plane 1 for each block mode. */
	float weight_low_values1[WEIGHTS_MAX_DECIMATION_MODES][TUNE_MAX_ANGULAR_QUANT + 1];/** @brief The low weight value in plane 1 for each quant level and decimation mode. */
	float weight_high_values1[WEIGHTS_MAX_DECIMATION_MODES][TUNE_MAX_ANGULAR_QUANT + 1];/** @brief The high weight value in plane 1 for each quant level and decimation mode. */
	
	int8_t qwt_bitcounts[WEIGHTS_MAX_BLOCK_MODES];/** @brief The total bit storage needed for quantized weights for each block mode. */
	float qwt_errors[WEIGHTS_MAX_BLOCK_MODES];/** @brief The cumulative error for quantized weights for each block mode. */

	float errors_of_best_combination[WEIGHTS_MAX_BLOCK_MODES];/** @brief Error of the best encoding combination for each block mode. */
	uint8_t best_quant_levels[WEIGHTS_MAX_BLOCK_MODES];/** @brief The best color quant for each block mode. */
	uint8_t best_quant_levels_mod[WEIGHTS_MAX_BLOCK_MODES];/** @brief The best color quant for each block mode if modes are the same and we have spare bits. */
	uint8_t best_ep_formats[WEIGHTS_MAX_BLOCK_MODES][BLOCK_MAX_PARTITIONS];/** @brief The best endpoint format for each partition. */
};

struct endpoints
{
	unsigned int partition_count;
	float4 endpt0[BLOCK_MAX_PARTITIONS];
	float4 endpt1[BLOCK_MAX_PARTITIONS];
};

struct endpoints_and_weights
{
	bool is_constant_weight_error_scale;
	endpoints ep;
	float weights[BLOCK_MAX_TEXELS];
	float weight_error_scale[BLOCK_MAX_TEXELS];
};


struct partition_metrics
{
	float4 avg;
	float4 dir;
};

__device__ const partition_info* get_partition_table(const block_size_descriptor* bsd, unsigned int partition_count)
{
	if (partition_count == 1)
	{
		partition_count = 5;
	}
	unsigned int index = (partition_count - 2) * BLOCK_MAX_PARTITIONINGS;
	return bsd->partitionings + index;
}

__device__ const partition_info* get_partition_info(const block_size_descriptor* bsd, unsigned int partition_count, unsigned int index)
{
	unsigned int packed_index = 0;
	if (partition_count >= 2)
	{
		packed_index = bsd->partitioning_packed_index[partition_count - 2][index];
	}

	const partition_info* result = &get_partition_table(bsd, partition_count)[packed_index];
	return result;
}

__device__ const decimation_mode& get_decimation_mode(const block_size_descriptor* bsd,unsigned int decimation_mode)
{
	return bsd->decimation_modes[decimation_mode];
}

const decimation_info& get_decimation_info(const block_size_descriptor* bsd, unsigned int decimation_mode)
{
	return bsd->decimation_tables[decimation_mode];
}

__device__ bool is_ref_1plane(const decimation_mode& dm, quant_method max_weight_quant)
{
	uint16_t mask = static_cast<uint16_t>((1 << (max_weight_quant + 1)) - 1);
	return (dm.refprec_1plane & mask) != 0;
}

__device__ unsigned int get_quant_level(quant_method method)
{
	switch (method)
	{
	case QUANT_2:   return   2;
	case QUANT_3:   return   3;
	case QUANT_4:   return   4;
	case QUANT_5:   return   5;
	case QUANT_6:   return   6;
	case QUANT_8:   return   8;
	case QUANT_10:  return  10;
	case QUANT_12:  return  12;
	case QUANT_16:  return  16;
	case QUANT_20:  return  20;
	case QUANT_24:  return  24;
	case QUANT_32:  return  32;
	case QUANT_40:  return  40;
	case QUANT_48:  return  48;
	case QUANT_64:  return  64;
	case QUANT_80:  return  80;
	case QUANT_96:  return  96;
	case QUANT_128: return 128;
	case QUANT_160: return 160;
	case QUANT_192: return 192;
	case QUANT_256: return 256;
	}

	// Unreachable - the enum is fully described
	return 0;
}

__device__ float4 normalize_safe(float4 a)
{
	float length = a.x * a.x + a.y * a.y + a.z * a.z+ a.w * a.w;
	if (length != 0.0f)
	{
		float inv_sqr_length = 1.0 / sqrt(length);
		return float4(a.x * inv_sqr_length, a.y * inv_sqr_length, a.z * inv_sqr_length, a.w * inv_sqr_length);
	}

	float val = 0.577350258827209473f;
	return float4(val, val, val, 0.0f);
}
#endif