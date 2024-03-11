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
	float rcp_stepsize = 1.0;
	for (unsigned int sp = 0; sp < max_angular_steps; sp++)
	{
		float minidx(128.0f);
		float maxidx(-128.0f);
		float errval = 0.0;
		float cut_low_weight_err = 0.0;
		float cut_high_weight_err = 0.0;
		float offset = offsets[sp];
		
		for (unsigned int j = 0; j < weight_count; j++)
		{
			float sval = dec_weight_ideal_value[j] * rcp_stepsize - offset;
			float svalrte = round(sval);
			float diff = sval - svalrte;
			errval += diff * diff;

			// Reset tracker on min hit
			minidx = minidx < svalrte ? svalrte : minidx;
			cut_low_weight_err = svalrte < minidx ? 0.0 : cut_low_weight_err;

			// Accumulate on min hit
			float accum = cut_low_weight_err + float(1.0f) - float(2.0f) * diff;
			cut_low_weight_err = (svalrte == minidx) ? accum : cut_low_weight_err;

			// Reset tracker on max hit
			maxidx = svalrte > maxidx ? svalrte : maxidx;
			cut_high_weight_err = svalrte > maxidx ? 0.0 : cut_high_weight_err;

			// Accumulate on max hit
			accum = cut_high_weight_err + float(1.0f) + float(2.0f) * diff;
			cut_high_weight_err = (svalrte == maxidx) ? accum : cut_high_weight_err;
		}

		// Write out min weight and weight span; clamp span to a usable range
		int span = int(maxidx - minidx + 1.0f);
		span = std::min(span, int(max_quant_steps + 3));
		span = std::max(span, 2);
		lowest_weight[sp] = minidx;
		weight_span[sp] = span;

		// The cut_(lowest/highest)_weight_error indicate the error that results from  forcing
		// samples that should have had the weight value one step (up/down).
		float ssize = 1.0f / rcp_stepsize;
		float errscale = ssize * ssize;
		error[sp] = errval * errscale;
		cut_low_weight_error[sp] = cut_low_weight_err * errscale;
		cut_high_weight_error[sp] = cut_high_weight_err * errscale;

		rcp_stepsize = rcp_stepsize + 1.0;
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
	compute_lowest_and_highest_weight(weight_count, dec_weight_ideal_value,
		max_angular_steps, max_quant_steps,
		angular_offsets, lowest_weight, weight_span, error,
		cut_low_weight_error, cut_high_weight_error);


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