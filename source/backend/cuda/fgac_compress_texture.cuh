#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define FLOAT_38 1e38f

#define BYTES_PER_DESTINATION_BLOCK 16 //128 bits

#define BLOCK_PIXEL_NUM (8 * 8)

#define BLOCK_DIM_WIDTH 8
#define BLOCK_DIM_HEIGHT 8

#define TEXELS_PER_BLOCK (8 * 8)
#define TEXELS_PER_BLOCK4 (8 * 8 * 4)


#define MAX_WEIGHTS_PER_BLOCK 64

typedef struct
{
    int     error_block;            // 1 marks error block, 0 marks non-error-block.
    int     block_mode;             // 0 to 2047. Negative value marks constant-color block (-1: FP16, -2:UINT16)
    int     partition_count;        // 1 to 4; Zero marks a constant-color block.
    int     partition_index;        // 0 to 1023
    int     color_formats[4];       // color format for each endpoint color pair.
    int     color_formats_matched;  // color format for all endpoint pairs are matched.
    int     color_values[4][12];    // quantized endpoint color pairs.
    int     color_quantization_level;
    uint8_t plane1_weights[MAX_WEIGHTS_PER_BLOCK];  // quantized and decimated weights
    uint8_t plane2_weights[MAX_WEIGHTS_PER_BLOCK];
    int     plane2_color_component;  // color component for the secondary plane of weights
    int     constant_color[4];       // constant-color, as FP16 or UINT16. Used for constant-color blocks only.
} symbolic_compressed_block;

typedef struct
{
    float4 orig_data[TEXELS_PER_BLOCK];   // original input data

    float red_min, red_max;
    float green_min, green_max;
    float blue_min, blue_max;
    float alpha_min, alpha_max;
} imageblock;
