#ifndef __YOLO11_HEAD_HPP__
#define __YOLO11_HEAD_HPP__
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "img_precess.hpp"
#include "det_structs.hpp"

#if USE_ENQUEUEV3
#include "trt_runner_v3.hpp"
#else
#include "trt_runner_v2.hpp"
#endif

class Yolo11Head {
public:
    struct Result {
        std::vector<Bbox2D> bbox_vec;
        float process_time;
    };

public:
    Yolo11Head()
    {}

public:
    int32_t Initialize(const std::string& model, const int& img_h, const int& img_w);
    int32_t ReadClsNames(const std::string& filename);
    void Finalize(void);
    void Process(const cv::Mat& input_cvmat, Result& result);
    
private:
#if USE_ENQUEUEV3
    std::unique_ptr<TrtRunnerV3> trt_runner_;
#else
    std::unique_ptr<TrtRunner> trt_runner_;
#endif
    cudaStream_t stream_;

private:
    std::unique_ptr<ImgPrecess> imgpcs_kernel_;
    PreParam pre_param_;

};

#endif