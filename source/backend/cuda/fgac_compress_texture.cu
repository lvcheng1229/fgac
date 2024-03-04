
#ifndef _FGAC_COMPRESS_TEXTURE_CU_
#define _FGAC_COMPRESS_TEXTURE_CU_



__gloabl__ void testKernel(uchar4* outputData, int width, int height, cudaTextureObject_t tex)
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = (float)x - (float)width / 2;
	float v = (float)y - (float)height / 2;

	u /= (float)width;
	v /= (float)height;


	// read from texture and write to global memory
	outputData[y * width + x] = tex2D<uchar4>(tex, u + 0.5f, v + 0.5f);
}

extern "C" void testKernel(dim3 gridSize, dim3 blockSize,uchar4 * outputData, int width, int height, cudaTextureObject_t tex)
{
	testKernel << <gridSize, blockSize,0 >> > (outputData, width, height, tex);
}



#endif