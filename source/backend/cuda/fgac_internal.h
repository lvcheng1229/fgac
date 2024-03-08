#pragma once
#include "fgac_compress_texture.h"
void init_partition_tables(
	block_size_descriptor& bsd,
	bool can_omit_partitionings,
	unsigned int partition_count_cutoff);