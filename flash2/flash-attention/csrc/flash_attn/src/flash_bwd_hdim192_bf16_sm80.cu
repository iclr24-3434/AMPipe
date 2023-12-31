// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<cutlass::bfloat16_t, 192>(Flash_bwd_params &params, cudaStream_t stream, const int offset, const bool configure) {
    run_mha_bwd_hdim192<cutlass::bfloat16_t>(params, stream, offset, configure);
}