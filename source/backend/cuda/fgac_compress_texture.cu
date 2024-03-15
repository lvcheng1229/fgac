#ifndef _FGAC_COMPRESS_TEXTURE_CU_
#define _FGAC_COMPRESS_TEXTURE_CU_

//#include "fgac_internal.cuh"
//#include "fgac_compress_texture.h"
//#include "fgac_compress_symbolic.cuh"
//#include "fgac_image.cuh"
//#include "fgac_symbolic_physical.cuh"

// note:
// 1. don't support hdr compression for now
// 2. only support u8 format

//__global__ void GPUEncodeKernel(uint8_t* outputData, cudaTextureObject_t tex, fgac_contexti* ctx)
//{
//	// calculate normalized texture coordinates
//	uint32_t global_index_x = blockIdx.x * blockDim.x + threadIdx.x;
//	uint32_t global_index_y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	const block_size_descriptor bsd = ctx->bsd;
//
//	image_block blk;
//
//	// block size e.g. 8x8
//	int block_x = bsd.xdim;
//	int block_y = bsd.ydim;
//	blk.texel_count = block_x * block_y;
//	blk.channel_weight = make_float4(ctx->config.cw_r_weight, ctx->config.cw_g_weight, ctx->config.cw_b_weight, ctx->config.cw_a_weight);
//
//	// image size
//	int dim_x = ctx->dim_x;
//	int dim_y = ctx->dim_y;
//
//	// the number of the blocks of each raw
//	int xblocks = (dim_x + block_x - 1) / block_x;
//	int yblocks = (dim_y + block_y - 1) / block_y;
//
//	int offset = (( yblocks + global_index_y) * xblocks + global_index_x) * 16;
//
//	if (global_index_x >= xblocks || global_index_y >= dim_y)
//	{
//		return;
//	}
//
//	uint2 start_pix_pos = make_uint2(global_index_x, global_index_y) * make_uint2(block_x, block_y);
//
//	uint8_t* dstData = outputData + offset;
//	compression_working_buffers tmpBuf;
//	load_image_block_fast_ldr(&blk, start_pix_pos, tex, ctx);
//	//compress_block(ctx,&blk, dstData,&tmpBuf);
//
//	return;
//}

/*__global__ void GPUDecodeKernel(uint8_t* compressedData, uint8_t* decompressedData, fgac_contexti* ctx)
{
#if !COMPRESS_ONLY
	uint32_t global_index_x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t global_index_y = blockIdx.y * blockDim.y + threadIdx.y;

	const block_size_descriptor bsd = ctx->bsd;

	int block_x = bsd.xdim;
	int block_y = bsd.ydim;

	int dim_x = ctx->dim_x;
	int dim_y = ctx->dim_y;

	int xblocks = (dim_x + block_x - 1) / block_x;
	int yblocks = (dim_y + block_y - 1) / block_y;

	int offset = ((yblocks + global_index_y) * xblocks + global_index_x) * 16;
	uint8_t* srcData = compressedData + offset;

	symbolic_compressed_block scb;
	physical_to_symbolic(&bsd, srcData, &scb);
#endif

}*/




//extern "C" void GPUEncodeKernel(dim3 gridSize, dim3 blockSize, uint8_t * outputData, cudaTextureObject_t tex, fgac_contexti * ctx)
//{
//	//helloCUDA << <gridSize, blockSize,0 >> > (1.2345f);
//	GPUEncodeKernel << <gridSize, blockSize, 0 >> > (outputData,tex, ctx);
//}
//
//extern "C" void GPUDecodeKernel(dim3 gridSize, dim3 blockSize, uint8_t* compressedData, uint8_t* decompressedData, fgac_contexti* ctx)
//{
//	GPUDecodeKernel << <gridSize, blockSize, 0 >> > (compressedData, decompressedData, ctx);
//}
#endif