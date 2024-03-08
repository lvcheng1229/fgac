#ifndef _FGAC_SYMBOLIC_PHYSICAL_CUH_
#define _FGAC_SYMBOLIC_PHYSICAL_CUH_
#include "fgac_internal.cuh"

//todo: read bits optimize

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
	int block_mode = read_bits(11, 0, pcb);
	if ((block_mode & 0x1FF) == 0x1FC) // constatnt block
	{
		return;
	}


}
#endif