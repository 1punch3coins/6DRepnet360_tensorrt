#ifndef __6D_REPNET360_HPP__
#define __6D_REPNET360_HPP__
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "img_precess.hpp"

#if USE_ENQUEUEV3
#include "trt_runner_v3.hpp"
#else
#include "trt_runner_v2.hpp"
#endif

class SixdRepnet360 {
public:
    struct Result {
        std::vector<std::array<float, 3>> angle_vec;
        float process_time;
    };

public:
    SixdRepnet360()
    {}

public:
    int32_t Initialize(const std::string& model, const int& img_h, const int& img_w);
    void Finalize(void);
    void Process(const cv::Mat& input_cvmat, const std::vector<Crop>& src_crops, Result& result);
    void DrawPoseAxis(cv::Mat& cvmat, const Result& res, const std::vector<Crop>& crop_vec);
    
private:
#if USE_ENQUEUEV3
    std::unique_ptr<TrtRunnerV3> trt_runner_;
#else
    std::unique_ptr<TrtRunner> trt_runner_;
#endif
    cudaStream_t stream_;
    std::vector<std::string> cls_names_;

private:
    std::unique_ptr<ImgPrecess> imgpcs_kernel_;
    PreParam pre_param_;
    float conf_thres_;
    float nms_iou_thres_;
};

#endif