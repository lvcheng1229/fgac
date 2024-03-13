#pragma once
#include "fgac_compress_texture.h"
void prepare_precomputed_tables(block_size_descriptor& bsd);
void prepare_quant_mode_table(block_size_descriptor& bsd);
void prepare_color_unquant_to_uquant_tables(block_size_descriptor& bsd); 
void prepare_color_uquant_to_scrambled_pquant_tables(block_size_descriptor& bsd);