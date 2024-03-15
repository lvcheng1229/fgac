#ifndef _FGAC_DEVICE_HOST_COMMON_H_
#define _FGAC_DEVICE_HOST_COMMON_H_
#include <stdint.h>
#include <vector_types.h>

#define BLOCK_MAX_TEXELS (8*8)
#define WEIGHTS_MAX_BLOCK_MODES 2048
#define BLOCK_MAX_WEIGHTS 64

enum quant_method
{
	QUANT_2 = 0,
	QUANT_3 = 1,
	QUANT_4 = 2,
	QUANT_5 = 3,
	QUANT_6 = 4,
	QUANT_8 = 5,
	QUANT_10 = 6,
	QUANT_12 = 7,
	QUANT_16 = 8,
	QUANT_20 = 9,
	QUANT_24 = 10,
	QUANT_32 = 11,

	QUANT_40 = 12,
	QUANT_48 = 13,
	QUANT_64 = 14,
	QUANT_80 = 15,
	QUANT_96 = 16,
	QUANT_128 = 17,
	QUANT_160 = 18,
	QUANT_192 = 19,
	QUANT_256 = 20
};

struct fgac_config
{
	float cw_sum_weight;
	float tune_db_limit;
};

struct block_size_descriptor
{
	uint8_t xdim; // The block x dimension
	uint8_t ydim; // The block y dimension

	uint8_t texel_count; // The block total texel count e.g. 8*8

	uint32_t block_mode_count_1plane_selected;
	block_mode block_modes[WEIGHTS_MAX_BLOCK_MODES];
};

struct fgac_contexti
{
	block_size_descriptor bsd;
	fgac_config config;
};

#endif

