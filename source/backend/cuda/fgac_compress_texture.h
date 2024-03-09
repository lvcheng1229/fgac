#ifndef _FGAC_COMPRESS_TEXTURE_H_
#define _FGAC_COMPRESS_TEXTURE_H_
#include <stdint.h>

#define BLOCK_MAX_TEXELS (8 * 8)
#define BLOCK_MAX_PARTITIONS 4
#define BLOCK_MAX_PARTITIONINGS 1024

#define SYM_BTYPE_ERROR 0
#define SYM_BTYPE_CONST_U16 2

#define TUNE_MIN_SEARCH_MODE0 0.85
#define BLOCK_BAD_BLOCK_MODE 0xFFFFu
#define WEIGHTS_MAX_BLOCK_MODES 2048 // block mode has 10 bit, that is to say, we have 2^10 possible solution
#define WEIGHTS_MAX_DECIMATION_MODES 87

#define WEIGHTS_TEXEL_SUM 16.0f

#define BLOCK_MAX_WEIGHTS 64
#define BLOCK_MIN_WEIGHT_BITS 24
#define BLOCK_MAX_WEIGHT_BITS 96
#define BLOCK_MAX_KMEANS_TEXELS 64 //The maximum number of texels used during partition selection for texel clustering

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

struct decimation_info
{
	uint8_t texel_count;
	uint8_t max_texel_weight_count;
	uint8_t weight_count;
	uint8_t weight_x;
	uint8_t weight_y;
	uint8_t weight_z;
	uint8_t texel_weight_count[BLOCK_MAX_TEXELS];
	uint8_t texel_weights_tr[4][BLOCK_MAX_TEXELS];
	uint8_t texel_weight_contribs_int_tr[4][BLOCK_MAX_TEXELS];
	float texel_weight_contribs_float_tr[4][BLOCK_MAX_TEXELS];
	uint8_t weight_texel_count[BLOCK_MAX_WEIGHTS];
	uint8_t weight_texels_tr[BLOCK_MAX_TEXELS][BLOCK_MAX_WEIGHTS];
	float weights_texel_contribs_tr[BLOCK_MAX_TEXELS][BLOCK_MAX_WEIGHTS];
	float texel_contrib_for_weight[BLOCK_MAX_TEXELS][BLOCK_MAX_WEIGHTS];
};

struct decimation_mode
{
	int8_t maxprec_1plane;
	int8_t maxprec_2planes;
	uint16_t refprec_1plane;
	uint16_t refprec_2planes;

	inline void set_ref_1plane(quant_method weight_quant) { refprec_1plane |= (1 << weight_quant); }
	inline void set_ref_2plane(quant_method weight_quant) { refprec_2planes |= static_cast<uint16_t>(1 << weight_quant); }
};

struct dt_init_working_buffers
{
	uint8_t weight_count_of_texel[BLOCK_MAX_TEXELS];
	uint8_t grid_weights_of_texel[BLOCK_MAX_TEXELS][4];
	uint8_t weights_of_texel[BLOCK_MAX_TEXELS][4];

	uint8_t texel_count_of_weight[BLOCK_MAX_WEIGHTS];
	uint8_t texels_of_weight[BLOCK_MAX_WEIGHTS][BLOCK_MAX_TEXELS];
	uint8_t texel_weights_of_weight[BLOCK_MAX_WEIGHTS][BLOCK_MAX_TEXELS];
};

struct partition_info
{
	uint16_t partition_count;/** @brief The number of partitions in this partitioning. */
	uint16_t partition_index;/** @brief The index (seed) of this partitioning. */
	uint8_t partition_texel_count[BLOCK_MAX_PARTITIONS];//brief The number of texels in each partition.
	uint8_t partition_of_texel[BLOCK_MAX_TEXELS];/** @brief The partition of each texel in the block. */
	uint8_t texels_of_partition[BLOCK_MAX_PARTITIONS][BLOCK_MAX_TEXELS];/** @brief The list of texels in each partition. */
};

struct block_mode
{
	uint16_t mode_index;
	uint8_t decimation_mode;
	uint8_t quant_mode;
	uint8_t weight_bits;
	uint8_t is_dual_plane : 1;

	inline quant_method get_weight_quant_mode() const { return static_cast<quant_method>(this->quant_mode); }
};

struct block_size_descriptor
{
	uint8_t xdim; // The block x dimension
	uint8_t ydim; // The block y dimension

	uint8_t texel_count; // The block total texel count e.g. 8*8

	uint16_t block_mode_packed_index[WEIGHTS_MAX_BLOCK_MODES]; // block mode indirect index array
	block_mode block_modes[WEIGHTS_MAX_BLOCK_MODES];
	decimation_info decimation_tables[WEIGHTS_MAX_DECIMATION_MODES];
	decimation_mode decimation_modes[WEIGHTS_MAX_DECIMATION_MODES];

	unsigned int block_mode_count_1plane_always;
	unsigned int block_mode_count_1plane_selected;
	unsigned int block_mode_count_1plane_2plane_selected;
	unsigned int block_mode_count_all;

	unsigned int partitioning_count_selected[BLOCK_MAX_PARTITIONS];
	unsigned int partitioning_count_all[BLOCK_MAX_PARTITIONS];

	unsigned int decimation_mode_count_always;
	unsigned int decimation_mode_count_selected;
	unsigned int decimation_mode_count_all;

	uint16_t partitioning_packed_index[3][BLOCK_MAX_PARTITIONINGS];
	uint8_t kmeans_texels[BLOCK_MAX_KMEANS_TEXELS]; // The active texels for k-means partition selection.

	partition_info partitionings[(3 * BLOCK_MAX_PARTITIONINGS) + 1];

	uint64_t coverage_bitmaps_2[BLOCK_MAX_PARTITIONINGS][2]; // The canonical 2-partition coverage pattern used during block partition search.
	uint64_t coverage_bitmaps_3[BLOCK_MAX_PARTITIONINGS][3]; 
	uint64_t coverage_bitmaps_4[BLOCK_MAX_PARTITIONINGS][4];
	
	//__device__ const block_mode& get_block_mode(unsigned int block_mode) const
	//{
	//	unsigned int packed_index = this->block_mode_packed_index[block_mode];
	//	return this->block_modes[packed_index];
	//}
	//
	//__device__ const decimation_info& get_decimation_info(unsigned int decimation_mode) const
	//{
	//	return this->decimation_tables[decimation_mode];
	//}
};

struct fgac_contexti
{
	fgac_config config;

	block_size_descriptor bsd;

	unsigned int dim_x; // The X dimension of the image e.g. 1024
	unsigned int dim_y; // The Y dimension of the image e.g. 1024


};

#endif
