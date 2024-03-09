#ifndef _FGAC_INTERNAL_CUH_
#define _FGAC_INTERNAL_CUH_

#define COMPRESS_ONLY 1

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
	uint8_t partition_count;
	uint16_t block_mode;
};

struct compression_working_buffers
{

};

__device__ partition_info* get_partition_table(block_size_descriptor* bsd, unsigned int partition_count)
{
	if (partition_count == 1)
	{
		partition_count = 5;
	}
	unsigned int index = (partition_count - 2) * BLOCK_MAX_PARTITIONINGS;
	return bsd->partitionings + index;
}

__device__ partition_info* get_partition_info(block_size_descriptor* bsd, unsigned int partition_count, unsigned int index)
{
	unsigned int packed_index = 0;
	if (partition_count >= 2)
	{
		packed_index = bsd->partitioning_packed_index[partition_count - 2][index];
	}

	partition_info* result = &get_partition_table(bsd, partition_count)[packed_index];
	return result;
}
#endif