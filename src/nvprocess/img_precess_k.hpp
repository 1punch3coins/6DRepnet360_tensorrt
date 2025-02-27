#pragma once
#include <iostream>
#include <vector>
#include "img_precess.hpp"

template<typename T0, typename T1>
cudaError_t launch_pre_kernel(const T0* d_mat_src, T1* d_mat_dst, const PreParam& param, void* stream_ptr);
template<typename T0, typename T1>
cudaError_t launch_pre_kernel_wto_shuffle(const T0* d_mat_src, T1* d_mat_dst, const PreParam& param, void* stream_ptr);

template<typename T0, typename T1>
cudaError_t launch_batched_pre_kernel(const std::vector<T0*>& d_mat_src_vec, T1* d_mat_dst, const std::vector<PreParam>& param_vec, void* stream_ptr);
template<typename T0, typename T1>
cudaError_t launch_batched_pre_kernel(const T0* d_mat_src, T1* d_mat_dst, const std::vector<PreParam>& param_vec, void* stream_ptr);

template<typename T0, typename T1>
cudaError_t launch_batched_pre_kernel_wto_shuffle(const std::vector<T0*>& d_mat_src_vec, T1* d_mat_dst, const std::vector<PreParam>& param_vec, void* stream_ptr);
template<typename T0, typename T1>
cudaError_t launch_batched_pre_kernel_wto_shuffle(const T0* d_mat_src, T1* d_mat_dst, const std::vector<PreParam>& param_vec, void* stream_ptr);