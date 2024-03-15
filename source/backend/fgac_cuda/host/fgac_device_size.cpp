#include "fgac_host_common.h"

void init_block_size_descriptor(
	unsigned int block_x_texels,
	unsigned int block_y_texels,
	block_size_descriptor& bsd
)
{
	bsd.xdim = block_x_texels;
	bsd.ydim = block_y_texels;
	bsd.texel_count = block_x_texels * block_y_texels;
}