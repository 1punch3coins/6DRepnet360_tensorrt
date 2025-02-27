#pragma once
#include <opencv2/opencv.hpp>
#include <vector_types.h>

enum class CropStyle: int32_t{
    SrcCropAll_DstCoverAll     = 0,
    SrcCropAll_DstEmbeddCnt    = 1,
    SrcEmbeddLower_DstCoverAll = 2,
    SrcCropLower_DstCoverAll   = 3
};

struct Crop{
    unsigned l;
    unsigned t;
    unsigned w;
    unsigned h;

    Crop() {}
    Crop(unsigned const& l_, unsigned const& t_, unsigned const& w_, unsigned const& h_):
        l(l_), t(t_), w(w_), h(h_)
    {}
};

struct PreParam{
    Crop src_crop;
    Crop dst_crop;
    int2 src_size;
    int2 dst_size;
    int src_step;
    int dst_step;
    float2 scale_inv;
    float3 mean;
    float3 norm_inv;
};

class ImgPrecess {
public:
    ImgPrecess()
    {}

public:
    void init_mem(int const& src_size, int const& batch_size=1);
    void dest();

public:
    void set_crop_attr(PreParam& param, int const& src_w, int const& src_h, int const& dst_w, int const& dst_h, CropStyle const& style);
    void set_dst_crop_attr(PreParam& param, Crop const& src_crop, int const& dst_w, int const& dst_h, CropStyle const& style);
    void convert_normalization_params(float3& mean, float3& norm);

public:
    void preprocess_crops(const std::vector<PreParam>& param_vec, bool is_color_shuffle, void* stream_ptr);
    void preprocess_cvmats(const std::vector<PreParam>& param_vec, bool is_color_shuffle, void* stream_ptr);

public:
    void* get_dev_in_ptr(const int& i) const {
        return d_mat_src_vec_[i];
    }
    void set_out_ptr(void* d_ptr) {
        d_mat_dst_ = d_ptr;
    }

private:
    std::vector<uint8_t*> d_mat_src_vec_;
    void* d_mat_dst_;
};