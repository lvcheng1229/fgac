#ifndef _FGAC_DEVICE_COMMON_H_
#define _FGAC_DEVICE_COMMON_H_
#include "../fgac_device_host_common.h"
#include "../common/helper_math.h"

struct block_mode
{
	uint16_t mode_index;
	uint8_t decimation_mode;
	uint8_t quant_mode;
	uint8_t weight_bits;
	uint8_t is_dual_plane : 1;
};

struct line4
{
	float4 a;
	float4 b;
};

struct endpoints
{
	float4 endpt0;
	float4 endpt1;
};

struct endpoints_and_weights
{
	endpoints ep;
	float weights[BLOCK_MAX_TEXELS];
};

struct symbolic_compressed_block
{
	endpoints_and_weights ei1;
};

struct compression_working_buffers
{
	endpoints_and_weights ei1;/** @brief Ideal endpoints and weights for plane 1. */
	float qwt_errors[WEIGHTS_MAX_BLOCK_MODES];
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
};

#endif