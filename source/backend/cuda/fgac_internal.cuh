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


struct image_block
{
	float data_r[BLOCK_MAX_TEXELS];
	float data_g[BLOCK_MAX_TEXELS];
	float data_b[BLOCK_MAX_TEXELS];
	float data_a[BLOCK_MAX_TEXELS];

	float4 data_min;
	float4 data_max;
};

//the unpacked content of a single physical compressed block
struct symbolic_compressed_block
{
	uint8_t block_type;
};

struct compression_working_buffers
{

};

#endif