#ifndef _FGAC_WEIGHTS_ALIGN_CUH_
#define _FGAC_WEIGHTS_ALIGN_CUH_

#include "fgac_internal.cuh"

__constant__ uint8_t steps_for_quant_level[12]{
	2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32
};

__device__ void compute_angular_offsets(
	const block_size_descriptor& bsd,
	unsigned int weight_count,
	const float* dec_weight_ideal_value,
	unsigned int max_angular_steps,
	float* offsets
) 
{
	int isample[BLOCK_MAX_WEIGHTS];

	for (unsigned int i = 0; i < weight_count; i ++)
	{
		float sample = dec_weight_ideal_value[i] * (SINCOS_STEPS - 1.0f) + float(12582912.0f);
		isample[i] = __float_as_int(sample) & int(SINCOS_STEPS - 1);
	}

	float mult = float(1.0f / (2.0f * PI));

	for (unsigned int i = 0; i < max_angular_steps; i ++)
	{
		float anglesum_x = 0;
		float anglesum_y = 0;

		for (unsigned int j = 0; j < weight_count; j++)
		{
			anglesum_x += bsd.cos_table[isample[j]][i];
			anglesum_y += bsd.sin_table[isample[j]][i];
		}

		float angle = atan2(anglesum_y, anglesum_x);
		float ofs = angle * mult;
		offsets[i] = ofs;
	}
}

__device__ void compute_lowest_and_highest_weight(
	unsigned int weight_count,
	const float* dec_weight_ideal_value,
	unsigned int max_angular_steps,
	unsigned int max_quant_steps,
	const float* offsets,
	float* lowest_weight,
	int* weight_span,
	float* error,
	float* cut_low_weight_error,
	float* cut_high_weight_error
)
{
	for (unsigned int sp = 0; sp < max_angular_steps; sp++)
	{
		for (unsigned int j = 0; j < weight_count; j++)
		{

		}
	}
}

__device__ void compute_angular_endpoints_for_quant_levels(
	const block_size_descriptor& bsd,
	unsigned int weight_count,
	const float* dec_weight_ideal_value,
	unsigned int max_quant_level,
	float low_value[TUNE_MAX_ANGULAR_QUANT + 1],
	float high_value[TUNE_MAX_ANGULAR_QUANT + 1]
) 
{
	unsigned int max_quant_steps = steps_for_quant_level[max_quant_level];
	unsigned int max_angular_steps = steps_for_quant_level[max_quant_level];
	float angular_offsets[ANGULAR_STEPS];

	compute_angular_offsets(bsd, weight_count, dec_weight_ideal_value, max_angular_steps, angular_offsets);

	float lowest_weight[ANGULAR_STEPS];
	int32_t weight_span[ANGULAR_STEPS];
	float error[ANGULAR_STEPS];
	float cut_low_weight_error[ANGULAR_STEPS];
	float cut_high_weight_error[ANGULAR_STEPS];
}


__device__ void compute_angular_endpoints_1plane(
	bool only_always,
	const block_size_descriptor& bsd,
	const float* dec_weight_ideal_value,
	unsigned int max_weight_quant,
	compression_working_buffers& tmpbuf
) 
{

}
#endif