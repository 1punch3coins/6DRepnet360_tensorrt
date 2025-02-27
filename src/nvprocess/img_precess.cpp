#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cu_marcos.h"
#include "img_precess.hpp"
#include "img_precess_k.hpp"

void ImgPrecess::set_crop_attr(PreParam& param, int const& src_w, int const& src_h, int const& dst_w, int const& dst_h, CropStyle const& style) {
    switch (style) {
        case CropStyle::SrcCropAll_DstCoverAll: { // retain
            param.src_crop.l = 0;
            param.src_crop.t = 0;
            param.src_crop.w = src_w;
            param.src_crop.h = src_h;
            param.dst_crop.l = 0;
            param.dst_crop.t = 0;
            param.dst_crop.w = dst_w;
            param.dst_crop.h = dst_h;
            param.scale_inv.x = static_cast<float>(param.src_crop.w) / param.dst_crop.w;   // note the conversion from integer to float
            param.scale_inv.y = static_cast<float>(param.src_crop.h) / param.dst_crop.h;
            break;
        }
        case CropStyle::SrcCropLower_DstCoverAll: {    // a very distinct one, shed 0.4 top part of src, disgard of ratio
            param.src_crop.l = 0;
            param.src_crop.t = src_h * 0.4f;
            param.src_crop.w = src_w;
            param.src_crop.h = src_h - param.src_crop.t;
            param.dst_crop.l = 0;
            param.dst_crop.t = 0;
            param.dst_crop.w = dst_w;
            param.dst_crop.h = dst_h;
            param.scale_inv.x = static_cast<float>(param.src_crop.w) / param.dst_crop.w;   // note the conversion from integer to float
            param.scale_inv.y = static_cast<float>(param.src_crop.h) / param.dst_crop.h;
            break;
        }
        case CropStyle::SrcEmbeddLower_DstCoverAll: {    // shed top part of src, retain width and make the crop ratio equals to model's input's
            float src_ratio = 1.0f * src_h / src_w;
            float dst_ratio = 1.0f * dst_h / dst_w;
            if (src_ratio > dst_ratio) {
                param.src_crop.w = src_w;
                param.src_crop.h = static_cast<int32_t>(src_w * dst_ratio);
                param.src_crop.l = 0;
                param.src_crop.t = src_h - param.src_crop.h;
            } else {
                param.src_crop.w = src_w;
                param.src_crop.h = src_h;
                param.src_crop.l = 0;
                param.src_crop.t = 0;
            }
            param.dst_crop.l = 0;
            param.dst_crop.t = 0;
            param.dst_crop.w = dst_w;
            param.dst_crop.h = dst_h;
            param.scale_inv.x = static_cast<float>(param.src_crop.w) / param.dst_crop.w;   // note the conversion from integer to float
            param.scale_inv.y = static_cast<float>(param.src_crop.h) / param.dst_crop.h;
            break;
        }
        case CropStyle::SrcCropAll_DstEmbeddCnt: {    // embedd src into dst's center, src's ratio not changed
            param.src_crop.l = 0;
            param.src_crop.t = 0;
            param.src_crop.w = src_w;
            param.src_crop.h = src_h;
            float src_ratio = 1.0f * src_w / src_h;
            float dst_ratio = 1.0f * dst_w / dst_h;
            if (src_ratio > dst_ratio) {
                // Use dst's width as base
                param.dst_crop.w = dst_w;
                param.dst_crop.h = static_cast<int32_t>(dst_w / src_ratio);
                param.dst_crop.l = 0;
                param.dst_crop.t = (dst_h - param.dst_crop.h) * 0.5f;
            } else {
                // Use dst's height as base
                param.dst_crop.h = dst_h;
                param.dst_crop.w = static_cast<int32_t>(dst_h * src_ratio);
                param.dst_crop.t = 0;
                param.dst_crop.l = (dst_w - param.dst_crop.w) * 0.5f;
            }
            param.scale_inv.x = static_cast<float>(param.src_crop.w) / param.dst_crop.w;   // note the conversion from integer to float
            param.scale_inv.y = static_cast<float>(param.src_crop.h) / param.dst_crop.h;
            break;
        }
        std::cerr << "crop style is not allowed" << std::endl;
    }
}

void ImgPrecess::set_dst_crop_attr(PreParam& param, Crop const& src_crop, int const& dst_w, int const& dst_h, CropStyle const& style) {
    switch (style) {
        case CropStyle::SrcCropAll_DstCoverAll: { // retain
            param.src_crop = src_crop;
            param.dst_crop.l = 0;
            param.dst_crop.t = 0;
            param.dst_crop.w = dst_w;
            param.dst_crop.h = dst_h;
            param.scale_inv.x = static_cast<float>(param.src_crop.w) / param.dst_crop.w;   // note the conversion from integer to float
            param.scale_inv.y = static_cast<float>(param.src_crop.h) / param.dst_crop.h;
            break;
        }
        case CropStyle::SrcCropLower_DstCoverAll: {    // a very distinct one, shed 0.4 top part of src, disgard of ratio
            param.src_crop.l = 0;
            param.src_crop.t = src_crop.h * 0.4f;
            param.src_crop.w = src_crop.w;
            param.src_crop.h = src_crop.h - param.src_crop.t;
            param.dst_crop.l = 0;
            param.dst_crop.t = 0;
            param.dst_crop.w = dst_w;
            param.dst_crop.h = dst_h;
            param.scale_inv.x = static_cast<float>(param.src_crop.w) / param.dst_crop.w;   // note the conversion from integer to float
            param.scale_inv.y = static_cast<float>(param.src_crop.h) / param.dst_crop.h;
            break;
        }
        case CropStyle::SrcEmbeddLower_DstCoverAll: {    // shed top part of src, retain width and make the crop ratio equals to model's input's
            float src_ratio = 1.0f * src_crop.h / src_crop.w;
            float dst_ratio = 1.0f * dst_h / dst_w;
            if (src_ratio > dst_ratio) {
                param.src_crop.w = src_crop.w;
                param.src_crop.h = static_cast<int32_t>(src_crop.w * dst_ratio);
                param.src_crop.l = src_crop.l;
                param.src_crop.t = src_crop.h - param.src_crop.h;
            } else {
                param.src_crop = src_crop;
            }
            param.dst_crop.l = 0;
            param.dst_crop.t = 0;
            param.dst_crop.w = dst_w;
            param.dst_crop.h = dst_h;
            param.scale_inv.x = static_cast<float>(param.src_crop.w) / param.dst_crop.w;   // note the conversion from integer to float
            param.scale_inv.y = static_cast<float>(param.src_crop.h) / param.dst_crop.h;
            break;
        }
        case CropStyle::SrcCropAll_DstEmbeddCnt: {    // embedd src into dst's center, src's ratio not changed
            param.src_crop = src_crop;
            float src_ratio = 1.0f * src_crop.w / src_crop.h;
            float dst_ratio = 1.0f * dst_w / dst_h;
            if (src_ratio > dst_ratio) {
                // Use dst's width as base
                param.dst_crop.w = dst_w;
                param.dst_crop.h = static_cast<int32_t>(dst_w / src_ratio);
                param.dst_crop.l = 0;
                param.dst_crop.t = (dst_h - param.dst_crop.h) * 0.5f;
            } else {
                // Use dst's height as base
                param.dst_crop.h = dst_h;
                param.dst_crop.w = static_cast<int32_t>(dst_h * src_ratio);
                param.dst_crop.t = 0;
                param.dst_crop.l = (dst_w - param.dst_crop.w) * 0.5f;
            }
            param.scale_inv.x = static_cast<float>(param.src_crop.w) / param.dst_crop.w;   // note the conversion from integer to float
            param.scale_inv.y = static_cast<float>(param.src_crop.h) / param.dst_crop.h;
            break;
        }
        std::cerr << "crop style is not allowed" << std::endl;
    }    
}

void ImgPrecess::convert_normalization_params(float3& mean, float3& norm) {
    // convert to speede up normalization: (((src/255) - mean)/norm)*scale ----> ((src - mean*255) / (255*norm))*scale ----> (src - mean*255) * (scale/(255*norm))
    mean.x *= 255;
    norm.x *= 255;
    norm.x = 1.0f / norm.x;
    mean.y *= 255;
    norm.y *= 255;
    norm.y = 1.0f / norm.y;
    mean.z *= 255;
    norm.z *= 255;
    norm.z = 1.0f / norm.z;
}

void ImgPrecess::init_mem(const int& src_size, const int& batch_size) {
    d_mat_src_vec_.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        CHECK_CUDA_FUNC(cudaMalloc((void**)&d_mat_src_vec_[i], src_size*3));
    }
}

void ImgPrecess::dest() {
    for (auto& d_mat_src: d_mat_src_vec_) {
        CHECK_CUDA_FUNC(cudaFree(d_mat_src));
    }
}

void ImgPrecess::preprocess_cvmats(const std::vector<PreParam>& param_vec, bool is_color_shuffle, void* stream_ptr) {
    if (is_color_shuffle) {
        CHECK_CUDA_FUNC(launch_batched_pre_kernel(d_mat_src_vec_, (float*)d_mat_dst_, param_vec, stream_ptr));
    } else {
        CHECK_CUDA_FUNC(launch_batched_pre_kernel_wto_shuffle(d_mat_src_vec_, (float*)d_mat_dst_, param_vec, stream_ptr));
    } 
}

void ImgPrecess::preprocess_crops(const std::vector<PreParam>& param_vec, bool is_color_shuffle, void* stream_ptr) {
    if (is_color_shuffle) {
        CHECK_CUDA_FUNC(launch_batched_pre_kernel(d_mat_src_vec_[0], (float*)d_mat_dst_, param_vec, stream_ptr));
    } else {
        CHECK_CUDA_FUNC(launch_batched_pre_kernel_wto_shuffle(d_mat_src_vec_[0], (float*)d_mat_dst_, param_vec, stream_ptr));
    }
}