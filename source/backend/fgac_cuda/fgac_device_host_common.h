#ifndef _FGAC_DEVICE_HOST_COMMON_H_
#define _FGAC_DEVICE_HOST_COMMON_H_
#include <stdint.h>
#include <vector_types.h>

#define BLOCK_MAX_TEXELS (8*8)


struct block_size_descriptor
{
	uint8_t xdim; // The block x dimension
	uint8_t ydim; // The block y dimension

	uint8_t texel_count; // The block total texel count e.g. 8*8
};

struct fgac_contexti
{
	block_size_descriptor bsd;
};

#endif

