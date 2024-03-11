#pragma once
#include <array>
#include <assert.h>
#include "fgac_internal.h"
#include "fgac_compress_texture.h"

struct ise_size
{
	/** @brief The scaling parameter. */
	uint8_t scale : 6;

	/** @brief The divisor parameter. */
	uint8_t divisor : 2;
};

// e.g. RANGE 80 = 4, 0, 1 : range = 5 * 2 ^ 4
// bit num = n * s + alignup( 7 * s / 3) n = 4 s = count
static const std::array<ise_size, 21> ise_sizes{ {
	{  1, 0 }, // QUANT_2
	{  8, 2 }, // QUANT_3
	{  2, 0 }, // QUANT_4
	{  7, 1 }, // QUANT_5
	{ 13, 2 }, // QUANT_6
	{  3, 0 }, // QUANT_8
	{ 10, 1 }, // QUANT_10
	{ 18, 2 }, // QUANT_12
	{  4, 0 }, // QUANT_16
	{ 13, 1 }, // QUANT_20
	{ 23, 2 }, // QUANT_24
	{  5, 0 }, // QUANT_32
	{ 16, 1 }, // QUANT_40
	{ 28, 2 }, // QUANT_48
	{  6, 0 }, // QUANT_64
	{ 19, 1 }, // QUANT_80
	{ 33, 2 }, // QUANT_96
	{  7, 0 }, // QUANT_128
	{ 22, 1 }, // QUANT_160
	{ 38, 2 }, // QUANT_192
	{  8, 0 }  // QUANT_256
} };

/* See header for documentation. */
unsigned int get_ise_sequence_bitcounts(
	unsigned int character_count,
	quant_method quant_level
) {
	if (static_cast<size_t>(quant_level) >= ise_sizes.size())
	{
		return 1024;
	}

	auto& entry = ise_sizes[quant_level];
	unsigned int divisor = (entry.divisor << 1) + 1;
	return (entry.scale * character_count + divisor - 1) / divisor;
}

static bool decode_block_mode_2d(
	unsigned int block_mode,
	unsigned int& x_weights,
	unsigned int& y_weights,
	bool& is_dual_plane,
	unsigned int& quant_mode,
	unsigned int& weight_bits
) 
{
	x_weights = 0;
	y_weights = 0;

	uint8_t R0 = (block_mode >> 4) & 1;
	uint8_t H = (block_mode >> 9) & 1;
	uint8_t D = (block_mode >> 10) & 1;
	uint8_t A = (block_mode >> 5) & 0x3;

	if ((block_mode & 3) != 0) // case bit 0 and bit 1 equal to R0 and R1
	{
		R0 |= (block_mode & 3) << 1; // R1 and R1
		unsigned int B = (block_mode >> 7) & 3;
		switch ((block_mode >> 2) & 3)
		{
		case 0:
			x_weights = B + 4;
			y_weights = A + 2;
			break;
		case 1:
			x_weights = B + 8;
			y_weights = A + 2;
			break;
		case 2:
			x_weights = A + 2;
			y_weights = B + 8;
			break;
		case 3:
			B &= 1;
			if (block_mode & 0x100)
			{
				x_weights = B + 2;
				y_weights = A + 2;
			}
			else
			{
				x_weights = A + 2;
				y_weights = B + 6;
			}
			break;
		}
	}
	else
	{
		R0 |= ((block_mode >> 2) & 3) << 1;
		if (((block_mode >> 2) & 3) == 0)
		{
			return false;
		}

		unsigned int B = (block_mode >> 9) & 3;
		switch ((block_mode >> 7) & 3)
		{
		case 0:
			x_weights = 12;
			y_weights = A + 2;
			break;
		case 1:
			x_weights = A + 2;
			y_weights = 12;
			break;
		case 2:
			x_weights = A + 6;
			y_weights = B + 6;
			D = 0;
			H = 0;
			break;
		case 3:
			switch ((block_mode >> 5) & 3)
			{
			case 0:
				x_weights = 6;
				y_weights = 10;
				break;
			case 1:
				x_weights = 10;
				y_weights = 6;
				break;
			case 2:
			case 3:
				return false;
			}
			break;
		}
	}

	unsigned int weight_count = x_weights * y_weights * (D + 1);
	quant_mode = (R0 - 2) + 6 * H;
	is_dual_plane = D != 0;
	weight_bits = get_ise_sequence_bitcounts(weight_count, static_cast<quant_method>(quant_mode));
	return (weight_count <= BLOCK_MAX_WEIGHTS && weight_bits >= BLOCK_MIN_WEIGHT_BITS && weight_bits <= BLOCK_MAX_WEIGHT_BITS);

}

static void init_decimation_info_2d(
	unsigned int x_texels,
	unsigned int y_texels,
	unsigned int x_weights,
	unsigned int y_weights,
	decimation_info& di,
	dt_init_working_buffers& wb
) {
	unsigned int texels_per_block = x_texels * y_texels;
	unsigned int weights_per_block = x_weights * y_weights;

	uint8_t max_texel_count_of_weight = 0;

	for (unsigned int i = 0; i < weights_per_block; i++)
	{
		wb.texel_count_of_weight[i] = 0;
	}

	for (unsigned int i = 0; i < texels_per_block; i++)
	{
		wb.weight_count_of_texel[i] = 0;
	}

	for (unsigned int y = 0; y < y_texels; y++)
	{
		for (unsigned int x = 0; x < x_texels; x++)
		{
			unsigned int texel = y * x_texels + x;

			unsigned int x_weight = (
				(
					(1024 + x_texels / 2) / (x_texels - 1)
					)
				*
				x
				*
				(x_weights - 1) + 32) >> 6;

			unsigned int y_weight = (((1024 + y_texels / 2) / (y_texels - 1)) * y * (y_weights - 1) + 32) >> 6;

			unsigned int x_weight_frac = x_weight & 0xF;
			unsigned int y_weight_frac = y_weight & 0xF;
			unsigned int x_weight_int = x_weight >> 4;
			unsigned int y_weight_int = y_weight >> 4;

			unsigned int qweight[4];
			qweight[0] = x_weight_int + y_weight_int * x_weights;
			qweight[1] = qweight[0] + 1;
			qweight[2] = qweight[0] + x_weights;
			qweight[3] = qweight[2] + 1;

			// Truncated-precision bilinear interpolation
			unsigned int prod = x_weight_frac * y_weight_frac;

			unsigned int weight[4];
			weight[3] = (prod + 8) >> 4;
			weight[1] = x_weight_frac - weight[3];
			weight[2] = y_weight_frac - weight[3];
			weight[0] = 16 - x_weight_frac - y_weight_frac + weight[3];

			for (unsigned int i = 0; i < 4; i++)
			{
				if (weight[i] != 0)
				{
					wb.grid_weights_of_texel[texel][wb.weight_count_of_texel[texel]] = static_cast<uint8_t>(qweight[i]);
					wb.weights_of_texel[texel][wb.weight_count_of_texel[texel]] = static_cast<uint8_t>(weight[i]);
					wb.weight_count_of_texel[texel]++;
					wb.texels_of_weight[qweight[i]][wb.texel_count_of_weight[qweight[i]]] = static_cast<uint8_t>(texel);
					wb.texel_weights_of_weight[qweight[i]][wb.texel_count_of_weight[qweight[i]]] = static_cast<uint8_t>(weight[i]);
					wb.texel_count_of_weight[qweight[i]]++;
					max_texel_count_of_weight = std::max(max_texel_count_of_weight, wb.texel_count_of_weight[qweight[i]]);
				}
			}
		}
	}

	uint8_t max_texel_weight_count = 0;
	for (unsigned int i = 0; i < texels_per_block; i++)
	{
		di.texel_weight_count[i] = wb.weight_count_of_texel[i];
		max_texel_weight_count = std::max(max_texel_weight_count, di.texel_weight_count[i]);

		for (unsigned int j = 0; j < wb.weight_count_of_texel[i]; j++)
		{
			di.texel_weight_contribs_int_tr[j][i] = wb.weights_of_texel[i][j];
			di.texel_weight_contribs_float_tr[j][i] = static_cast<float>(wb.weights_of_texel[i][j]) * (1.0f / WEIGHTS_TEXEL_SUM);
			di.texel_weights_tr[j][i] = wb.grid_weights_of_texel[i][j];
		}

		// Init all 4 entries so we can rely on zeros for vectorization
		for (unsigned int j = wb.weight_count_of_texel[i]; j < 4; j++)
		{
			di.texel_weight_contribs_int_tr[j][i] = 0;
			di.texel_weight_contribs_float_tr[j][i] = 0.0f;
			di.texel_weights_tr[j][i] = 0;
		}
	}

	di.max_texel_weight_count = max_texel_weight_count;

	for (unsigned int i = 0; i < weights_per_block; i++)
	{
		unsigned int texel_count_wt = wb.texel_count_of_weight[i];
		di.weight_texel_count[i] = static_cast<uint8_t>(texel_count_wt);

		for (unsigned int j = 0; j < texel_count_wt; j++)
		{
			uint8_t texel = wb.texels_of_weight[i][j];

			// Create transposed versions of these for better vectorization
			di.weight_texels_tr[j][i] = texel;
			di.weights_texel_contribs_tr[j][i] = static_cast<float>(wb.texel_weights_of_weight[i][j]);

			// Store the per-texel contribution of this weight for each texel it contributes to
			di.texel_contrib_for_weight[j][i] = 0.0f;
			for (unsigned int k = 0; k < 4; k++)
			{
				uint8_t dttw = di.texel_weights_tr[k][texel];
				float dttwf = di.texel_weight_contribs_float_tr[k][texel];
				if (dttw == i && dttwf != 0.0f)
				{
					di.texel_contrib_for_weight[j][i] = di.texel_weight_contribs_float_tr[k][texel];
					break;
				}
			}
		}

		// Initialize array tail so we can over-fetch with SIMD later to avoid loop tails
		// Match last texel in active lane in SIMD group, for better gathers
		uint8_t last_texel = di.weight_texels_tr[texel_count_wt - 1][i];
		for (unsigned int j = texel_count_wt; j < max_texel_count_of_weight; j++)
		{
			di.weight_texels_tr[j][i] = last_texel;
			di.weights_texel_contribs_tr[j][i] = 0.0f;
		}
	}

	di.texel_count = static_cast<uint8_t>(texels_per_block);
	di.weight_count = static_cast<uint8_t>(weights_per_block);
	di.weight_x = static_cast<uint8_t>(x_weights);
	di.weight_y = static_cast<uint8_t>(y_weights);
	di.weight_z = 1;
}

static inline uint64_t rotl(uint64_t val, int count)
{
	return (val << count) | (val >> (64 - count));
}

static void assign_kmeans_texels(
	block_size_descriptor& bsd
) {
	// Use all texels for kmeans on a small block
	if (bsd.texel_count <= BLOCK_MAX_KMEANS_TEXELS)
	{
		for (uint8_t i = 0; i < bsd.texel_count; i++)
		{
			bsd.kmeans_texels[i] = i;
		}

		return;
	}

	// Select a random subset of BLOCK_MAX_KMEANS_TEXELS for kmeans on a large block
	uint64_t rng_state[2];
	rng_state[0] = 0xfaf9e171cea1ec6bULL;
	rng_state[1] = 0xf1b318cc06af5d71ULL;

	// Initialize array used for tracking used indices
	bool seen[BLOCK_MAX_TEXELS];
	for (uint8_t i = 0; i < bsd.texel_count; i++)
	{
		seen[i] = false;
	}

	// Assign 64 random indices, retrying if we see repeats
	unsigned int arr_elements_set = 0;
	while (arr_elements_set < BLOCK_MAX_KMEANS_TEXELS)
	{
		uint64_t s0 = rng_state[0];
		uint64_t s1 = rng_state[1];
		uint64_t res = s0 + s1;
		s1 ^= s0;
		rng_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
		rng_state[1] = rotl(s1, 37);

		uint8_t texel = static_cast<uint8_t>(res);
		texel = texel % bsd.texel_count;
		if (!seen[texel])
		{
			bsd.kmeans_texels[arr_elements_set++] = texel;
			seen[texel] = true;
		}
	}
}

static void construct_dt_entry_2d(
	unsigned int x_texels,
	unsigned int y_texels,
	unsigned int x_weights,
	unsigned int y_weights,
	block_size_descriptor& bsd,
	dt_init_working_buffers& wb,
	unsigned int index
) {
	unsigned int weight_count = x_weights * y_weights;
	assert(weight_count <= BLOCK_MAX_WEIGHTS);

	bool try_2planes = (2 * weight_count) <= BLOCK_MAX_WEIGHTS;

	decimation_info& di = bsd.decimation_tables[index];
	init_decimation_info_2d(x_texels, y_texels, x_weights, y_weights, di, wb);

	int maxprec_1plane = -1;
	int maxprec_2planes = -1;
	for (int i = 0; i < 12; i++)
	{
		unsigned int bits_1plane = get_ise_sequence_bitcounts(weight_count, static_cast<quant_method>(i));
		if (bits_1plane >= BLOCK_MIN_WEIGHT_BITS && bits_1plane <= BLOCK_MAX_WEIGHT_BITS)
		{
			maxprec_1plane = i;
		}

		if (try_2planes)
		{
			unsigned int bits_2planes = get_ise_sequence_bitcounts(2 * weight_count, static_cast<quant_method>(i));
			if (bits_2planes >= BLOCK_MIN_WEIGHT_BITS && bits_2planes <= BLOCK_MAX_WEIGHT_BITS)
			{
				maxprec_2planes = i;
			}
		}
	}

	// At least one of the two should be valid ...
	assert(maxprec_1plane >= 0 || maxprec_2planes >= 0);
	bsd.decimation_modes[index].maxprec_1plane = static_cast<int8_t>(maxprec_1plane);
	bsd.decimation_modes[index].maxprec_2planes = static_cast<int8_t>(maxprec_2planes);
	bsd.decimation_modes[index].refprec_1plane = 0;
	bsd.decimation_modes[index].refprec_2planes = 0;
}

// we set can_omit_modes = true for compressor and set can_omit_modes = false for decompressor since we may decompress the image generated by other compressor
static void construct_block_size_descriptor_2d(
	unsigned int x_texels,
	unsigned int y_texels,
	bool can_omit_modes,
	float mode_cutoff,
	block_size_descriptor& bsd)
{
	static const unsigned int MAX_DMI = 8 * 16 + 8; // we only support 8x8 grid for now
	int decimation_mode_index[MAX_DMI];

	dt_init_working_buffers* wb = new dt_init_working_buffers;

	bsd.xdim = static_cast<uint8_t>(x_texels);
	bsd.ydim = static_cast<uint8_t>(y_texels);
	bsd.texel_count = static_cast<uint8_t>(x_texels * y_texels);

	for (unsigned int i = 0; i < MAX_DMI; i++)
	{
		decimation_mode_index[i] = -1;
	}

	// Construct the list of block formats referencing the decimation tables
	unsigned int packed_bm_idx = 0;
	unsigned int packed_dm_idx = 0;

	// Trackers
	unsigned int bm_counts[4]{ 0 };
	unsigned int dm_counts[4]{ 0 };

	for (unsigned int i = 0; i < WEIGHTS_MAX_BLOCK_MODES; i++)
	{
		bsd.block_mode_packed_index[i] = BLOCK_BAD_BLOCK_MODE;
	}

	// Iterate four times to build a usefully ordered list:
	//   - Pass 0 - keep selected single plane "always" block modes
	//   - Pass 1 - keep selected single plane "non-always" block modes
	//   - Pass 2 - keep select dual plane block modes
	//   - Pass 3 - keep everything else that's legal

	unsigned int limit = can_omit_modes ? 3 : 4;
	for (unsigned int j = 0; j < limit; j++)
	{
		for (unsigned int i = 0; i < WEIGHTS_MAX_BLOCK_MODES; i++)
		{
			// Skip modes we've already included in a previous pass
			if (bsd.block_mode_packed_index[i] != BLOCK_BAD_BLOCK_MODE)
			{
				continue;
			}

			// Decode parameters
			unsigned int x_weights;
			unsigned int y_weights;
			bool is_dual_plane;
			unsigned int quant_mode;
			unsigned int weight_bits;
			bool valid = decode_block_mode_2d(i, x_weights, y_weights, is_dual_plane, quant_mode, weight_bits);

			// Always skip invalid encodings for the current block size
			if (!valid || (x_weights > x_texels) || (y_weights > y_texels))
			{
				continue;
			}

			// Selectively skip dual plane encodings
			if (((j <= 1) && is_dual_plane) || (j == 2 && !is_dual_plane))
			{
				continue;
			}

			// Always skip encodings we can't physically encode based on
			// generic encoding bit availability
			if (is_dual_plane)
			{
				// This is the only check we need as only support 1 partition
				if ((109 - weight_bits) <= 0)
				{
					continue;
				}
			}
			else
			{
				// This is conservative - fewer bits may be available for > 1 partition
				if ((111 - weight_bits) <= 0)
				{
					continue;
				}
			}

			//if (j != 3)
			//{
			//	
			//}

			int decimation_mode = decimation_mode_index[y_weights * 16 + x_weights];
			if (decimation_mode < 0)
			{
				construct_dt_entry_2d(x_texels, y_texels, x_weights, y_weights, bsd, *wb, packed_dm_idx);
				decimation_mode_index[y_weights * 16 + x_weights] = packed_dm_idx;
				decimation_mode = packed_dm_idx;

				dm_counts[j]++;
				packed_dm_idx++;
			}

			auto& bm = bsd.block_modes[packed_bm_idx];

			bm.decimation_mode = static_cast<uint8_t>(decimation_mode);
			bm.quant_mode = static_cast<uint8_t>(quant_mode);
			bm.is_dual_plane = static_cast<uint8_t>(is_dual_plane);
			bm.weight_bits = static_cast<uint8_t>(weight_bits);
			bm.mode_index = static_cast<uint16_t>(i);

			auto& dm = bsd.decimation_modes[decimation_mode];
			if (is_dual_plane)
			{
				dm.set_ref_2plane(bm.get_weight_quant_mode());
			}
			else
			{
				dm.set_ref_1plane(bm.get_weight_quant_mode());
			}

			bsd.block_mode_packed_index[i] = static_cast<uint16_t>(packed_bm_idx);

			packed_bm_idx++;
			bm_counts[j]++;
		}
	}

	bsd.block_mode_count_1plane_always = bm_counts[0];
	bsd.block_mode_count_1plane_selected = bm_counts[0] + bm_counts[1];
	bsd.block_mode_count_1plane_2plane_selected = bm_counts[0] + bm_counts[1] + bm_counts[2];
	bsd.block_mode_count_all = bm_counts[0] + bm_counts[1] + bm_counts[2] + bm_counts[3];

	bsd.decimation_mode_count_always = dm_counts[0];
	bsd.decimation_mode_count_selected = dm_counts[0] + dm_counts[1] + dm_counts[2];
	bsd.decimation_mode_count_all = dm_counts[0] + dm_counts[1] + dm_counts[2] + dm_counts[3];

	for (unsigned int i = bsd.decimation_mode_count_all; i < WEIGHTS_MAX_DECIMATION_MODES; i++)
	{
		bsd.decimation_modes[i].maxprec_1plane = -1;
		bsd.decimation_modes[i].maxprec_2planes = -1;
		bsd.decimation_modes[i].refprec_1plane = 0;
		bsd.decimation_modes[i].refprec_2planes = 0;
	}

	assign_kmeans_texels(bsd);

	delete wb;
}

void prepare_angular_tables(block_size_descriptor& bsd)
{
	for (unsigned int i = 0; i < ANGULAR_STEPS; i++)
	{
		float angle_step = static_cast<float>(i + 1);

		for (unsigned int j = 0; j < SINCOS_STEPS; j++)
		{
			bsd.sin_table[j][i] = static_cast<float>(sinf((2.0f * PI / (SINCOS_STEPS - 1.0f)) * angle_step * static_cast<float>(j)));
			bsd.cos_table[j][i] = static_cast<float>(cosf((2.0f * PI / (SINCOS_STEPS - 1.0f)) * angle_step * static_cast<float>(j)));
		}
	}
}

void init_block_size_descriptor(
	unsigned int x_texels,
	unsigned int y_texels,
	unsigned int z_texels,
	bool can_omit_modes,
	unsigned int partition_count_cutoff,
	float mode_cutoff,
	block_size_descriptor& bsd
) 
{
	construct_block_size_descriptor_2d(x_texels, y_texels, can_omit_modes, mode_cutoff, bsd);
	init_partition_tables(bsd, can_omit_modes, partition_count_cutoff);
	prepare_angular_tables(bsd);
}