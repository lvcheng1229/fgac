#ifndef _FGAC_INT_SEQUENCE_CUH_
#define _FGAC_INT_SEQUENCE_CUH_

#include "fgac_internal.cuh"

#if !COMPRESS_ONLY
struct btq_count
{
	uint8_t bits : 6;/** @brief The number of bits. */
	uint8_t trits : 1;/** @brief The number of trits. */
	uint8_t quints : 1;/** @brief The number of quints. */
};

__constant__ btq_count btq_counts[32] 
{
	{ 1, 0, 0 }, // QUANT_2
	{ 0, 1, 0 }, // QUANT_3
	{ 2, 0, 0 }, // QUANT_4
	{ 0, 0, 1 }, // QUANT_5
	{ 1, 1, 0 }, // QUANT_6
	{ 3, 0, 0 }, // QUANT_8
	{ 1, 0, 1 }, // QUANT_10
	{ 2, 1, 0 }, // QUANT_12
	{ 4, 0, 0 }, // QUANT_16
	{ 2, 0, 1 }, // QUANT_20
	{ 3, 1, 0 }, // QUANT_24
	{ 5, 0, 0 }, // QUANT_32
	{ 3, 0, 1 }, // QUANT_40
	{ 4, 1, 0 }, // QUANT_48
	{ 6, 0, 0 }, // QUANT_64
	{ 4, 0, 1 }, // QUANT_80
	{ 5, 1, 0 }, // QUANT_96
	{ 7, 0, 0 }, // QUANT_128
	{ 5, 0, 1 }, // QUANT_160
	{ 6, 1, 0 }, // QUANT_192
	{ 8, 0, 0 }  // QUANT_256
};

struct ise_size
{
	/** @brief The scaling parameter. */
	uint8_t scale : 6;

	/** @brief The divisor parameter. */
	uint8_t divisor : 2;
};

__constant__ ise_size  ise_sizes[21] 
{
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
};

__device__ unsigned int read_bits(
	unsigned int bitcount,
	unsigned int bitoffset,
	const uint8_t* ptr
) {
	unsigned int mask = (1 << bitcount) - 1;
	ptr += bitoffset >> 3;
	bitoffset &= 7;
	unsigned int value = ptr[0] | (ptr[1] << 8);
	value >>= bitoffset;
	value &= mask;
	return value;
}

/* See header for documentation. */
__device__ unsigned int get_ise_sequence_bitcount(
	unsigned int character_count,
	quant_method quant_level
) {
	// Cope with out-of bounds values - input might be invalid
	if (static_cast<size_t>(quant_level) >= 21)
	{
		// Arbitrary large number that's more than an ASTC block can hold
		return 1024;
	}

	auto& entry = ise_sizes[quant_level];
	unsigned int divisor = (entry.divisor << 1) + 1;
	return (entry.scale * character_count + divisor - 1) / divisor;
}

__constant__ uint8_t quints_of_integer[128][3]{
	{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0},
	{4, 0, 0}, {0, 4, 0}, {4, 4, 0}, {4, 4, 4},
	{0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0},
	{4, 1, 0}, {1, 4, 0}, {4, 4, 1}, {4, 4, 4},
	{0, 2, 0}, {1, 2, 0}, {2, 2, 0}, {3, 2, 0},
	{4, 2, 0}, {2, 4, 0}, {4, 4, 2}, {4, 4, 4},
	{0, 3, 0}, {1, 3, 0}, {2, 3, 0}, {3, 3, 0},
	{4, 3, 0}, {3, 4, 0}, {4, 4, 3}, {4, 4, 4},
	{0, 0, 1}, {1, 0, 1}, {2, 0, 1}, {3, 0, 1},
	{4, 0, 1}, {0, 4, 1}, {4, 0, 4}, {0, 4, 4},
	{0, 1, 1}, {1, 1, 1}, {2, 1, 1}, {3, 1, 1},
	{4, 1, 1}, {1, 4, 1}, {4, 1, 4}, {1, 4, 4},
	{0, 2, 1}, {1, 2, 1}, {2, 2, 1}, {3, 2, 1},
	{4, 2, 1}, {2, 4, 1}, {4, 2, 4}, {2, 4, 4},
	{0, 3, 1}, {1, 3, 1}, {2, 3, 1}, {3, 3, 1},
	{4, 3, 1}, {3, 4, 1}, {4, 3, 4}, {3, 4, 4},
	{0, 0, 2}, {1, 0, 2}, {2, 0, 2}, {3, 0, 2},
	{4, 0, 2}, {0, 4, 2}, {2, 0, 4}, {3, 0, 4},
	{0, 1, 2}, {1, 1, 2}, {2, 1, 2}, {3, 1, 2},
	{4, 1, 2}, {1, 4, 2}, {2, 1, 4}, {3, 1, 4},
	{0, 2, 2}, {1, 2, 2}, {2, 2, 2}, {3, 2, 2},
	{4, 2, 2}, {2, 4, 2}, {2, 2, 4}, {3, 2, 4},
	{0, 3, 2}, {1, 3, 2}, {2, 3, 2}, {3, 3, 2},
	{4, 3, 2}, {3, 4, 2}, {2, 3, 4}, {3, 3, 4},
	{0, 0, 3}, {1, 0, 3}, {2, 0, 3}, {3, 0, 3},
	{4, 0, 3}, {0, 4, 3}, {0, 0, 4}, {1, 0, 4},
	{0, 1, 3}, {1, 1, 3}, {2, 1, 3}, {3, 1, 3},
	{4, 1, 3}, {1, 4, 3}, {0, 1, 4}, {1, 1, 4},
	{0, 2, 3}, {1, 2, 3}, {2, 2, 3}, {3, 2, 3},
	{4, 2, 3}, {2, 4, 3}, {0, 2, 4}, {1, 2, 4},
	{0, 3, 3}, {1, 3, 3}, {2, 3, 3}, {3, 3, 3},
	{4, 3, 3}, {3, 4, 3}, {0, 3, 4}, {1, 3, 4}
};

__constant__ uint8_t trits_of_integer[256][5]{
	{0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {2, 0, 0, 0, 0}, {0, 0, 2, 0, 0},
	{0, 1, 0, 0, 0}, {1, 1, 0, 0, 0}, {2, 1, 0, 0, 0}, {1, 0, 2, 0, 0},
	{0, 2, 0, 0, 0}, {1, 2, 0, 0, 0}, {2, 2, 0, 0, 0}, {2, 0, 2, 0, 0},
	{0, 2, 2, 0, 0}, {1, 2, 2, 0, 0}, {2, 2, 2, 0, 0}, {2, 0, 2, 0, 0},
	{0, 0, 1, 0, 0}, {1, 0, 1, 0, 0}, {2, 0, 1, 0, 0}, {0, 1, 2, 0, 0},
	{0, 1, 1, 0, 0}, {1, 1, 1, 0, 0}, {2, 1, 1, 0, 0}, {1, 1, 2, 0, 0},
	{0, 2, 1, 0, 0}, {1, 2, 1, 0, 0}, {2, 2, 1, 0, 0}, {2, 1, 2, 0, 0},
	{0, 0, 0, 2, 2}, {1, 0, 0, 2, 2}, {2, 0, 0, 2, 2}, {0, 0, 2, 2, 2},
	{0, 0, 0, 1, 0}, {1, 0, 0, 1, 0}, {2, 0, 0, 1, 0}, {0, 0, 2, 1, 0},
	{0, 1, 0, 1, 0}, {1, 1, 0, 1, 0}, {2, 1, 0, 1, 0}, {1, 0, 2, 1, 0},
	{0, 2, 0, 1, 0}, {1, 2, 0, 1, 0}, {2, 2, 0, 1, 0}, {2, 0, 2, 1, 0},
	{0, 2, 2, 1, 0}, {1, 2, 2, 1, 0}, {2, 2, 2, 1, 0}, {2, 0, 2, 1, 0},
	{0, 0, 1, 1, 0}, {1, 0, 1, 1, 0}, {2, 0, 1, 1, 0}, {0, 1, 2, 1, 0},
	{0, 1, 1, 1, 0}, {1, 1, 1, 1, 0}, {2, 1, 1, 1, 0}, {1, 1, 2, 1, 0},
	{0, 2, 1, 1, 0}, {1, 2, 1, 1, 0}, {2, 2, 1, 1, 0}, {2, 1, 2, 1, 0},
	{0, 1, 0, 2, 2}, {1, 1, 0, 2, 2}, {2, 1, 0, 2, 2}, {1, 0, 2, 2, 2},
	{0, 0, 0, 2, 0}, {1, 0, 0, 2, 0}, {2, 0, 0, 2, 0}, {0, 0, 2, 2, 0},
	{0, 1, 0, 2, 0}, {1, 1, 0, 2, 0}, {2, 1, 0, 2, 0}, {1, 0, 2, 2, 0},
	{0, 2, 0, 2, 0}, {1, 2, 0, 2, 0}, {2, 2, 0, 2, 0}, {2, 0, 2, 2, 0},
	{0, 2, 2, 2, 0}, {1, 2, 2, 2, 0}, {2, 2, 2, 2, 0}, {2, 0, 2, 2, 0},
	{0, 0, 1, 2, 0}, {1, 0, 1, 2, 0}, {2, 0, 1, 2, 0}, {0, 1, 2, 2, 0},
	{0, 1, 1, 2, 0}, {1, 1, 1, 2, 0}, {2, 1, 1, 2, 0}, {1, 1, 2, 2, 0},
	{0, 2, 1, 2, 0}, {1, 2, 1, 2, 0}, {2, 2, 1, 2, 0}, {2, 1, 2, 2, 0},
	{0, 2, 0, 2, 2}, {1, 2, 0, 2, 2}, {2, 2, 0, 2, 2}, {2, 0, 2, 2, 2},
	{0, 0, 0, 0, 2}, {1, 0, 0, 0, 2}, {2, 0, 0, 0, 2}, {0, 0, 2, 0, 2},
	{0, 1, 0, 0, 2}, {1, 1, 0, 0, 2}, {2, 1, 0, 0, 2}, {1, 0, 2, 0, 2},
	{0, 2, 0, 0, 2}, {1, 2, 0, 0, 2}, {2, 2, 0, 0, 2}, {2, 0, 2, 0, 2},
	{0, 2, 2, 0, 2}, {1, 2, 2, 0, 2}, {2, 2, 2, 0, 2}, {2, 0, 2, 0, 2},
	{0, 0, 1, 0, 2}, {1, 0, 1, 0, 2}, {2, 0, 1, 0, 2}, {0, 1, 2, 0, 2},
	{0, 1, 1, 0, 2}, {1, 1, 1, 0, 2}, {2, 1, 1, 0, 2}, {1, 1, 2, 0, 2},
	{0, 2, 1, 0, 2}, {1, 2, 1, 0, 2}, {2, 2, 1, 0, 2}, {2, 1, 2, 0, 2},
	{0, 2, 2, 2, 2}, {1, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 0, 2, 2, 2},
	{0, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {2, 0, 0, 0, 1}, {0, 0, 2, 0, 1},
	{0, 1, 0, 0, 1}, {1, 1, 0, 0, 1}, {2, 1, 0, 0, 1}, {1, 0, 2, 0, 1},
	{0, 2, 0, 0, 1}, {1, 2, 0, 0, 1}, {2, 2, 0, 0, 1}, {2, 0, 2, 0, 1},
	{0, 2, 2, 0, 1}, {1, 2, 2, 0, 1}, {2, 2, 2, 0, 1}, {2, 0, 2, 0, 1},
	{0, 0, 1, 0, 1}, {1, 0, 1, 0, 1}, {2, 0, 1, 0, 1}, {0, 1, 2, 0, 1},
	{0, 1, 1, 0, 1}, {1, 1, 1, 0, 1}, {2, 1, 1, 0, 1}, {1, 1, 2, 0, 1},
	{0, 2, 1, 0, 1}, {1, 2, 1, 0, 1}, {2, 2, 1, 0, 1}, {2, 1, 2, 0, 1},
	{0, 0, 1, 2, 2}, {1, 0, 1, 2, 2}, {2, 0, 1, 2, 2}, {0, 1, 2, 2, 2},
	{0, 0, 0, 1, 1}, {1, 0, 0, 1, 1}, {2, 0, 0, 1, 1}, {0, 0, 2, 1, 1},
	{0, 1, 0, 1, 1}, {1, 1, 0, 1, 1}, {2, 1, 0, 1, 1}, {1, 0, 2, 1, 1},
	{0, 2, 0, 1, 1}, {1, 2, 0, 1, 1}, {2, 2, 0, 1, 1}, {2, 0, 2, 1, 1},
	{0, 2, 2, 1, 1}, {1, 2, 2, 1, 1}, {2, 2, 2, 1, 1}, {2, 0, 2, 1, 1},
	{0, 0, 1, 1, 1}, {1, 0, 1, 1, 1}, {2, 0, 1, 1, 1}, {0, 1, 2, 1, 1},
	{0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 1, 1, 1, 1}, {1, 1, 2, 1, 1},
	{0, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, {2, 2, 1, 1, 1}, {2, 1, 2, 1, 1},
	{0, 1, 1, 2, 2}, {1, 1, 1, 2, 2}, {2, 1, 1, 2, 2}, {1, 1, 2, 2, 2},
	{0, 0, 0, 2, 1}, {1, 0, 0, 2, 1}, {2, 0, 0, 2, 1}, {0, 0, 2, 2, 1},
	{0, 1, 0, 2, 1}, {1, 1, 0, 2, 1}, {2, 1, 0, 2, 1}, {1, 0, 2, 2, 1},
	{0, 2, 0, 2, 1}, {1, 2, 0, 2, 1}, {2, 2, 0, 2, 1}, {2, 0, 2, 2, 1},
	{0, 2, 2, 2, 1}, {1, 2, 2, 2, 1}, {2, 2, 2, 2, 1}, {2, 0, 2, 2, 1},
	{0, 0, 1, 2, 1}, {1, 0, 1, 2, 1}, {2, 0, 1, 2, 1}, {0, 1, 2, 2, 1},
	{0, 1, 1, 2, 1}, {1, 1, 1, 2, 1}, {2, 1, 1, 2, 1}, {1, 1, 2, 2, 1},
	{0, 2, 1, 2, 1}, {1, 2, 1, 2, 1}, {2, 2, 1, 2, 1}, {2, 1, 2, 2, 1},
	{0, 2, 1, 2, 2}, {1, 2, 1, 2, 2}, {2, 2, 1, 2, 2}, {2, 1, 2, 2, 2},
	{0, 0, 0, 1, 2}, {1, 0, 0, 1, 2}, {2, 0, 0, 1, 2}, {0, 0, 2, 1, 2},
	{0, 1, 0, 1, 2}, {1, 1, 0, 1, 2}, {2, 1, 0, 1, 2}, {1, 0, 2, 1, 2},
	{0, 2, 0, 1, 2}, {1, 2, 0, 1, 2}, {2, 2, 0, 1, 2}, {2, 0, 2, 1, 2},
	{0, 2, 2, 1, 2}, {1, 2, 2, 1, 2}, {2, 2, 2, 1, 2}, {2, 0, 2, 1, 2},
	{0, 0, 1, 1, 2}, {1, 0, 1, 1, 2}, {2, 0, 1, 1, 2}, {0, 1, 2, 1, 2},
	{0, 1, 1, 1, 2}, {1, 1, 1, 1, 2}, {2, 1, 1, 1, 2}, {1, 1, 2, 1, 2},
	{0, 2, 1, 1, 2}, {1, 2, 1, 1, 2}, {2, 2, 1, 1, 2}, {2, 1, 2, 1, 2},
	{0, 2, 2, 2, 2}, {1, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 1, 2, 2, 2}
};

__device__ void decode_ise(
	quant_method quant_level,
	unsigned int character_count,
	const uint8_t* input_data,
	uint8_t* output_data,
	unsigned int bit_offset
)
{
	// Note: due to how the trit/quint-block unpacking is done in this function, we may write more
	// temporary results than the number of outputs. The maximum actual number of results is 64 bit,
	// but we keep 4 additional character_count of padding.
	uint8_t results[68];
	uint8_t tq_blocks[22]{ 0 }; // Trit-blocks or quint-blocks, must be zeroed

	unsigned int bits = btq_counts[quant_level].bits;
	unsigned int trits = btq_counts[quant_level].trits;
	unsigned int quints = btq_counts[quant_level].quints;

	unsigned int lcounter = 0;
	unsigned int hcounter = 0;

	// Collect bits for each element, as well as bits for any trit-blocks and quint-blocks.
	for (unsigned int i = 0; i < character_count; i++)
	{
		results[i] = static_cast<uint8_t>(read_bits(bits, bit_offset, input_data));
		bit_offset += bits;

		if (trits)
		{
			static const uint8_t bits_to_read[5]{ 2, 2, 1, 2, 1 };
			static const uint8_t block_shift[5]{ 0, 2, 4, 5, 7 };
			static const uint8_t next_lcounter[5]{ 1, 2, 3, 4, 0 };
			static const uint8_t hcounter_incr[5]{ 0, 0, 0, 0, 1 };
			unsigned int tdata = read_bits(bits_to_read[lcounter], bit_offset, input_data);
			bit_offset += bits_to_read[lcounter];
			tq_blocks[hcounter] |= tdata << block_shift[lcounter];
			hcounter += hcounter_incr[lcounter];
			lcounter = next_lcounter[lcounter];
		}

		// bits / 3
		if (quints)
		{
			static const uint8_t bits_to_read[3]{ 3, 2, 2 };
			static const uint8_t block_shift[3]{ 0, 3, 5 };
			static const uint8_t next_lcounter[3]{ 1, 2, 0 };
			static const uint8_t hcounter_incr[3]{ 0, 0, 1 };
			unsigned int tdata = read_bits(bits_to_read[lcounter], bit_offset, input_data);
			bit_offset += bits_to_read[lcounter];
			tq_blocks[hcounter] |= tdata << block_shift[lcounter];
			hcounter += hcounter_incr[lcounter];
			lcounter = next_lcounter[lcounter];
		}
	}

	// Unpack trit-blocks or quint-blocks as needed
	if (trits)
	{
		unsigned int trit_blocks = (character_count + 4) / 5;
		for (unsigned int i = 0; i < trit_blocks; i++)
		{
			const uint8_t* tritptr = trits_of_integer[tq_blocks[i]];
			results[5 * i] |= tritptr[0] << bits;
			results[5 * i + 1] |= tritptr[1] << bits;
			results[5 * i + 2] |= tritptr[2] << bits;
			results[5 * i + 3] |= tritptr[3] << bits;
			results[5 * i + 4] |= tritptr[4] << bits;
		}
	}

	if (quints)
	{
		unsigned int quint_blocks = (character_count + 2) / 3;
		for (unsigned int i = 0; i < quint_blocks; i++)
		{
			const uint8_t* quintptr = quints_of_integer[tq_blocks[i]];
			results[3 * i] |= quintptr[0] << bits;
			results[3 * i + 1] |= quintptr[1] << bits;
			results[3 * i + 2] |= quintptr[2] << bits;
		}
	}

	for (unsigned int i = 0; i < character_count; i++)
	{
		output_data[i] = results[i];
	}
}
#endif
#endif