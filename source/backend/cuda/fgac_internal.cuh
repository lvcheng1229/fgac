#ifndef _FGAC_INTERNAL_CUH_
#define _FGAC_INTERNAL_CUH_

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "fgac_compress_texture.h"

#define BLOCK_MAX_TEXELS (12 * 12)

#define SYM_BTYPE_CONST_U16 2
#define TUNE_MIN_SEARCH_MODE0 0.85
#define WEIGHTS_MAX_BLOCK_MODES 2048 // block mode has 10 bit, that is to say, we have 2^10 possible solution

struct image_block
{
	float data_r[BLOCK_MAX_TEXELS];
	float data_g[BLOCK_MAX_TEXELS];
	float data_b[BLOCK_MAX_TEXELS];
	float data_a[BLOCK_MAX_TEXELS];

	float4 data_min;
	float4 data_max;
};

struct compression_working_buffers
{

};

#endif