typedef struct
{
	float4 rgba_weights;
}error_weighting_params;

typedef struct
{
	uint32_t m_srcTexWidth;
	uint32_t m_srcTexHeight;

	uint32_t m_xDim; // block dimension x, e.g. 8x8
	uint32_t m_yDim; // block dimension u, e.g. 8x8

	uint32_t m_blkNumX;
	uint32_t m_blkNumY;

	float m_compressQuality;

	error_weighting_params ewp;
}SAstcEncoderInfo;
