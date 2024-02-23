// todo:
// 1. static const offset array 
// 2. find a better way to compute the end points
// 3. unroll
// 4. find the best block mode

#define GROUP_SIZE 8
#define MAX_MIP_NUM 14
#define BLOCK_SIZE (8 * 8) //TODO:
#define IS_NORMALMAP 0
#define HAS_ALPHA 1
#define SMALL_VALUE 1e-6

SamplerState gSamPointWarp : register(s0, space1000);
SamplerState gSamLinearWarp : register(s4, space1000);
SamplerState gSamLinearClamp : register(s5, space1000);

Texture2D<float4> inputTexture             : register(t0);
RWStructuredBuffer<uint4> outputBuffer     : register(u0);

cbuffer fgacConstBuffer : register(b0)
{
    uint mipStartBlockIndex[MAX_MIP_NUM];
    uint m_groupNumX;
    uint m_blockNum;
    uint m_mipsNum;
    uint m_mip0TexWidth;
    uint m_mip0TexHeight;
    uint2 m_blockSize;
}

uint PackTexel(float4 texel)
{
    uint4 packedTexel = (uint4)texel;
    return (packedTexel.x << 0) | (packedTexel.x << 8) | (packedTexel.x << 16) | (packedTexel.x << 24);
}

float4 UnpackTexel(uint packedTexel)
{
	uint4 unpackedTexel;
	unpackedTexel.x = (packedTexel >> 0) & 0xff;
	unpackedTexel.y = (packedTexel >> 8) & 0xff;
	unpackedTexel.z = (packedTexel >> 16) & 0xff;
	unpackedTexel.w = (packedTexel >> 24) & 0xff;
	return (float4)unpackedTexel;
}

// TODO: static const offset array
float4 GetTexel(uint3 blockPixelStartPos, float2 mipInvSize, uint subBlockIndex)
{
    uint y = subBlockIndex / m_blockSize.x;
    uint x = subBlockIndex - y * m_blockSize.x;
    uint2 pixelPos = uint2(x, y) + blockPixelStartPos.xy;
    float2 pixelUV = pixePos * mipInvSize;
    float4 texel = inputTexture.SampleLevel(gSamPointWarp, pixelUV, blockPixelStartPos.z);
#if IS_NORMALMAP
    texel.z = 1.0f;
    texel.w = 1.0f;
#endif
    return texel;
}

void FindMinMaxFromBlock(const uint texelBlock[BLOCK_SIZE], float4 pixelMean, float4 maxDir, out float4 endPoint0, out float4 endPoint1)
{
    float minProjT =  1e30f;
    float maxProjT = -1e30f;

    for(uint index = 0; index < BLOCK_SIZE; index++)
    {
        float4 texel = UnpackTexel(texelBlock[index]);
        texel -= pixelMean;
        float projT = dot(texel,maxDir);
        minProjT = min(minProjT,projT);
        maxProjT = min(maxProjT,projT);
    }

    endPoint0 = clamp(maxDir * minProjT + pixelMean, 0.0f, 255.0f);
	endPoint1 = clamp(maxDir * maxProjT + pixelMean, 0.0f, 255.0f);

    //
    float4 e0u = round(endPoint0);
	float4 e1u = round(endPoint1);
	if (e0u.x + e0u.y + e0u.z > e1u.x + e1u.y + e1u.z)
	{
		swap(endPoint0, endPoint1);
	}
#if !HAS_ALPHA
	endPoint0.a = 255.0f;
	endPoint1.a = 255.0f;
#endif
}

// TODO: not a good solution, must modify this function
void MaxAccumulationPixelDirection(uint3 blockPixelStartPos, float2 mipInvSize, inout uint texelBlock[BLOCK_SIZE], out float4 endPoint0, out float4 endPoint1)
{
    float4 pixelMean = 0;

    //TODO
	for (uint index = 0; index < BLOCK_SIZE; index++)
    {
        float4 texel = GetTexel(blockPixelStartPos, mipInvSize, index);
        texelBlock[index] = PackTexel(texel);
        pixelMean += texel;
    }

    pixelMean /= BLOCK_SIZE;

    float4 sumr = 0;
    float4 sumg = 0;
    float4 sumb = 0;
    float4 suma = 0;

    for (uint index = 0; index < BLOCK_SIZE; index++)
    {
        float4 texel = UnpackTexel(texelBlock[index]);
        float4 dt = texel - pixelMean;
        sumr += (dt.x > 0) ? dt : 0;
		sumg += (dt.y > 0) ? dt : 0;
		sumb += (dt.z > 0) ? dt : 0;
		suma += (dt.w > 0) ? dt : 0;
    }

    float dotr = dot(sumr, sumr);
    float dotg = dot(sumg, sumg);
    float dotb = dot(sumb, sumb);
    float dota = dot(suma, suma);

    //todo: remove these dynamic branch
    float maxDot = dotr;
    float maxDir = sumr;

    if(dotg > maxdot)
    {
        maxDir = sumg;
        maxDot = dotg;
    }

    if(dotb > maxdot)
    {
        maxDir = sumb;
        maxDot = dotb;
    }

#if HAS_ALPHA
    if(dota > maxdot)
    {
        maxDir = suma;
        maxDot = dota;
    }
#endif

    float lenMaxDir = length(maxDir);
	maxDir = (lenMaxDir < SMALL_VALUE) ? maxDir : normalize(maxDir);

    FindMinMaxFromBlock(texelBlock, pixelMean, maxDir, endPoint0, endPoint1);
}

uint4 EncodeBlock(uint3 blockPixelStartPos, float2 mipInvSize)
{
    uint  texelBlock[BLOCK_SIZE];
    float4 endPoint0;
    float4 endPoint1;
    MaxAccumulationPixelDirection(blockPixelStartPos,mipInvSize,texelBlock,endPoint0,endPoint1);

    // TODO:
#if HAS_ALPHA
	uint4 bestBlockMode = uint4(QUANT_6, QUANT_256, 6, 7);
#else
	uint4 bestBlockMode = uint4(QUANT_12, QUANT_256, 12, 7);
#endif
}

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void MainCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint rowBlockNum = m_groupNumX * GROUP_SIZE;
    uint blockIndex = dispatchThreadID.y * rowBlockNum + dispatchThreadID.x;

    if(blockIndex >= m_blockNum){return;}
    
    //TODO:
    uint mipLevel = 0;
    for(uint index = 0; index < m_mipsNum; index++) 
    {
        if(blockIndex < mipStartBlockIndex[index])
        {
            mipLevel = blockIndex - 1;
            break;
        }
    }

    uint2 curMipSize = uint2(m_mip0TexWidth >> mipLevel, m_mip0TexHeight >> mipLevel);
    uint2 curMipBlockNum = uint2(
        (curMipSize.x + m_blockSize.x - 1) / m_blockSize.x,
        (curMipSize.y + m_blockSize.y - 1) / m_blockSize.y);

    uint mipBlockIndex = blockIndex - mipStartBlockIndex[mipLevel];

    uint3 blockPixelStartPos;
    blockPixelStartPos.y = mipBlockIndex / curMipBlockNum.x;
    blockPixelStartPos.x = mipBlockIndex - blockPixelStartPos.y * curMipBlockNum.x;
    blockPixelStartPos.z = mipLevel;

    blockPixelStartPos.xy *= m_blockSize;

    float2 mipInvSize = float2(1.0f / curMipSize.x, 1.0f / curMipSize.y);
    outputBuffer[blockIndex] = EncodeBlock(blockPixelStartPos,mipInvSize);
}