#ifndef _FGAC_COMPRESS_SYMBOLIC_H_
#define _FGAC_COMPRESS_SYMBOLIC_H_
#include "fgac_decvice_common.cuh"
__device__ void compress_block(fgac_contexti* ctx, image_block* blk, uint8_t pcb[16], compression_working_buffers* tmpbuf)
{
	if ((blk->data_min.x == blk->data_max.x) &&
		(blk->data_min.y == blk->data_max.y) &&
		(blk->data_min.z == blk->data_max.z) &&
		(blk->data_min.w == blk->data_max.w))
	{
		//todo:
	}

	// search 1 partition and 1 plane
	float errorval = compress_symbolic_block_for_partition_1plane(
		ctx->config, bsd, *blk, error_threshold, scb, *tmpbuf, QUANT_32);
}
#endif