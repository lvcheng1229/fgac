#ifndef _FGAC_COMPRESS_TEXTURE_H_
#define _FGAC_COMPRESS_TEXTURE_H_

struct fgac_config
{
	float cw_r_weight;
	float cw_g_weight;
	float cw_b_weight;
	float cw_a_weight;

	float cw_sum_weight; //fgac

	float tune_db_limit; //The dB threshold for stopping block search(-dblimit).

	/**
	* @brief The config enable for the mode0 fast-path search.
	*
	* If this is set to TUNE_MIN_TEXELS_MODE0 or higher then the early-out fast mode0
	* search is enabled. This option is ineffective for 3D block sizes.
	*/
	float tune_search_mode0_enable;
};

struct block_size_descriptor
{
	uint8_t xdim; // The block x dimension
	uint8_t ydim; // The block y dimension

	uint8_t texel_count; // The block total texel count e.g. 8*8
};

struct fgac_contexti
{
	fgac_config config;

	block_size_descriptor bsd;

	unsigned int dim_x; // The X dimension of the image e.g. 1024
	unsigned int dim_y; // The Y dimension of the image e.g. 1024
};

#endif
