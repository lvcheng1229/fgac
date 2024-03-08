#ifndef _FGAC_COMPRESS_TEXTURE_H_
#define _FGAC_COMPRESS_TEXTURE_H_
#include <stdint.h>

#define BLOCK_BAD_BLOCK_MODE 0xFFFFu
#define WEIGHTS_MAX_BLOCK_MODES 2048 // block mode has 10 bit, that is to say, we have 2^10 possible solution

#define BLOCK_MAX_WEIGHTS 64
#define BLOCK_MIN_WEIGHT_BITS 24
#define BLOCK_MAX_WEIGHT_BITS 96

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
	//unsigned int block_x; // 8x8 ...
	//unsigned int block_y;

	float cw_r_weight;
	float cw_g_weight;
	float cw_b_weight;
	float cw_a_weight;

	float cw_sum_weight; //fgac

	float tune_db_limit; //The dB threshold for stopping block search(-dblimit).
};

struct block_mode
{
	uint16_t mode_index;
	uint8_t decimation_mode;
	uint8_t quant_mode;
	uint8_t weight_bits;
	uint8_t is_dual_plane : 1;
};

struct block_size_descriptor
{
	uint8_t xdim; // The block x dimension
	uint8_t ydim; // The block y dimension

	uint8_t texel_count; // The block total texel count e.g. 8*8

	uint16_t block_mode_packed_index[WEIGHTS_MAX_BLOCK_MODES]; // block mode indirect index array
	block_mode block_modes[WEIGHTS_MAX_BLOCK_MODES];
};

struct fgac_contexti
{
	fgac_config config;

	block_size_descriptor bsd;

	unsigned int dim_x; // The X dimension of the image e.g. 1024
	unsigned int dim_y; // The Y dimension of the image e.g. 1024


};

#endif
