#include "fgac_internal.h"

static uint8_t select_partition(
	int seed,
	int x,
	int y,
	int z,
	int partition_count,
	bool small_block
) 
{

}
static bool generate_one_partition_info_entry(
	block_size_descriptor& bsd,
	unsigned int partition_count,
	unsigned int partition_index,
	unsigned int partition_remap_index,
	partition_info& pi
) 
{
	int texels_per_block = bsd.texel_count;
	bool small_block = texels_per_block < 32;

	uint8_t* partition_of_texel = pi.partition_of_texel;

	int texel_idx = 0;
	int counts[BLOCK_MAX_PARTITIONS]{ 0 };

	for (unsigned int y = 0; y < bsd.ydim; y++)
	{
		for (unsigned int x = 0; x < bsd.xdim; x++)
		{
			uint8_t part = select_partition(partition_index, x, y, 1, partition_count, small_block);
		}
	}
}

void init_partition_tables(block_size_descriptor& bsd, bool can_omit_partitionings, unsigned int partition_count_cutoff)
{
	partition_info* par_tab2 = bsd.partitionings;
	partition_info* par_tab3 = par_tab2 + BLOCK_MAX_PARTITIONINGS;
	partition_info* par_tab4 = par_tab3 + BLOCK_MAX_PARTITIONINGS;
	partition_info* par_tab1 = par_tab4 + BLOCK_MAX_PARTITIONINGS;
}