#ifndef _FGAC_SYMBOLIC_PHYSICAL_CUH_
#define _FGAC_SYMBOLIC_PHYSICAL_CUH_
#include "fgac_internal.cuh"
#include "fgac_integer_sequence.cuh"

#if !COMPRESS_ONLY
__device__ int bitrev8(int p)
{
	p = ((p & 0x0F) << 4) | ((p >> 4) & 0x0F);
	p = ((p & 0x33) << 2) | ((p >> 2) & 0x33);
	p = ((p & 0x55) << 1) | ((p >> 1) & 0x55);
	return p;
}

__device__ int read_bits(int bitcount, int bitoffset, const uint8_t* ptr)
{
	int mask = (1 << bitcount) - 1;
	ptr += bitoffset >> 3;
	bitoffset &= 7;
	int value = ptr[0] | (ptr[1] << 8);
	value >>= bitoffset;
	value &= mask;
	return value;
}

__device__ void physical_to_symbolic(const block_size_descriptor* bsd, const uint8_t pcb[16], symbolic_compressed_block* scb)
{
	int block_mode_var = read_bits(11, 0, pcb);
	if ((block_mode_var & 0x1FF) == 0x1FC) // constatnt block
	{
		return;
	}

	unsigned int packed_index = bsd->block_mode_packed_index[block_mode_var];
	if (packed_index == BLOCK_BAD_BLOCK_MODE)
	{
		scb->block_type = SYM_BTYPE_ERROR;
		return;
	}

	//const block_mode& bm = bsd->get_block_mode(block_mode_var);
	//const auto& di = bsd->get_decimation_info(bm.decimation_mode);

	block_mode bm ;
	decimation_info di;

	int weight_count = di.weight_count;

	quant_method weight_quant_method = static_cast<quant_method>(bm.quant_mode);
	int is_dual_plane = bm.is_dual_plane;

	int real_weight_count = is_dual_plane ? 2 * weight_count : weight_count;

	int partition_count = read_bits(2, 11, pcb) + 1;

	scb->block_mode = static_cast<uint16_t>(block_mode_var);
	scb->partition_count = static_cast<uint8_t>(partition_count);

	uint8_t bswapped[16];
	for (int i = 0; i < 16; i++)
	{
		bswapped[i] = static_cast<uint8_t>(bitrev8(pcb[15 - i]));
	}

	int bits_for_weights = get_ise_sequence_bitcount(real_weight_count, weight_quant_method);

	uint8_t indices[64];
	decode_ise(weight_quant_method, real_weight_count, bswapped, indices, 0);

	//int below_weights_pos = 128 - bits_for_weights;
	//if (is_dual_plane)
	//{
	//	for (int i = 0; i < weight_count; i++)
	//	{
	//		scb.weights[i] = qat.unscramble_and_unquant_map[indices[2 * i]];
	//		scb.weights[i + WEIGHTS_PLANE2_OFFSET] = qat.unscramble_and_unquant_map[indices[2 * i + 1]];
	//	}
	//}
	//else
	//{
	//
	//}
}
#endif
#endif