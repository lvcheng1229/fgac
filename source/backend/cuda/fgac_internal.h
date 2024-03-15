#pragma once
#include "fgac_compress_texture.h"
//void init_partition_tables(
//	block_size_descriptor& bsd,
//	bool can_omit_partitionings,
//	unsigned int partition_count_cutoff);

void init_block_size_descriptor(
	unsigned int x_texels,
	unsigned int y_texels,
	unsigned int z_texels,
	bool can_omit_modes,
	unsigned int partition_count_cutoff,
	float mode_cutoff,
	block_size_descriptor& bsd
);