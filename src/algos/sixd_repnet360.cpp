#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <array>
#include "algo_logger.hpp"
#include "sixd_repnet360.hpp"

static bool kDynamicBatchSize = true;

// model's input's and output's meta
using Tt = NetMeta::TensorType;
static constexpr int32_t kInputTensorNum = 1;
static constexpr int32_t kOutputTensorNum = 1;
static constexpr int32_t kBatchSize = 16;
static constexpr std::array<const char*, kInputTensorNum> sInputNameList = {"input"};
static constexpr std::array<const bool, kInputTensorNum> iInputNchwList = {true};
static constexpr std::array<const Tt, kInputTensorNum> kInputTypeList = {Tt::kTypeFloat32};
static constexpr std::array<const char*, kOutputTensorNum> sOutputNameList = {"output"};
static constexpr std::array<const bool, kOutputTensorNum> iOutputNlcList = {true};
static constexpr std::array<const Tt, kOutputTensorNum> kOutputTypeList = {Tt::kTypeFloat32};
// used for pre_process
static constexpr float kMeanList[] = {0.485f, 0.456f, 0.406f};
static constexpr float kNormList[] = {0.229f, 0.224f, 0.225f};
// used for post_process and net output meta check
static constexpr int32_t kOutputLen = 3;
static constexpr int32_t kOutputChannelNum = 3;
static constexpr float kExpandRatio = 1.4f;
// used for logger
static constexpr const char* sIdentifier = {"6DRepnet360"};
static constexpr int32_t kLogInfoSize = 100;
static auto logger = algoLogger::logger;

int32_t SixdRepnet360::Initialize(const std::string& model, const int& img_h, const int& img_w) {
    // set net meta from pre_config
    LOG_INFO(logger, sIdentifier, "initializing...");
    NetMeta* p_meta = new NetMeta(kInputTensorNum, kOutputTensorNum);
    for (int32_t i = 0; i < kInputTensorNum; i++) {
        p_meta->AddInputTensorMeta(sInputNameList[i], kInputTypeList[i], iInputNchwList[i]);
    }
    for (int32_t i = 0; i < kOutputTensorNum; i++) {
        p_meta->AddOutputTensorMeta(sOutputNameList[i], kOutputTypeList[i], iOutputNlcList[i]);
    }
#if USE_ENQUEUEV3
    trt_runner_.reset(TrtRunnerV3::Create());
#else
    trt_runner_.reset(TrtRunner::Create());
#endif

    // create model and set net meta values from engine
    if (trt_runner_->InitEngine(model)) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "engine creation failed");
    }
    LOG_INFO(logger, sIdentifier, "model's engine creation completed");
    if (trt_runner_->InitMeta(p_meta)) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "net meta initialization failed");
    }
    LOG_INFO(logger, sIdentifier, "net meta initialization completed");
    if (trt_runner_->InitBatchSize(kDynamicBatchSize, kBatchSize)) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "batch size initialization failed");
    }
    LOG_INFO(logger, sIdentifier, "batch size initialization completed");
    if (trt_runner_->InitMem()) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "model's memory initialization failed");
    }
    LOG_INFO(logger, sIdentifier, "model's memory initialization completed");

    // check output tensor meta
    if (trt_runner_->GetOutChannelNum(sOutputNameList[0]) != kOutputChannelNum) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "output channel size mismatched");
    }
    if (trt_runner_->GetOutLen(sOutputNameList[0]) != kOutputLen) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "output len mismatched");
    }
    LOG_INFO(logger, sIdentifier, "output meta check completed");

    // create cuda stream
    CHECK_CUDA_FUNC(cudaStreamCreate(&stream_));
    
    // set image preprocess kernel
    const int32_t net_in_h = trt_runner_->GetInHeight(sInputNameList[0]);
    const int32_t net_in_w = trt_runner_->GetInWidth(sInputNameList[0]);
    imgpcs_kernel_ = std::make_unique<ImgPrecess>();
    pre_param_.src_size.x = img_w;     // input image's size
    pre_param_.src_size.y = img_h;
    pre_param_.dst_size.x = net_in_w;  // output mat(i.e model's input)'s size
    pre_param_.dst_size.y = net_in_h;
    pre_param_.src_step   = img_w;     // input mat's step(it should be each row's byte size; here for kernel's simplicity, it is each row's element num)
    pre_param_.dst_step   = net_in_w;  // output mat's step
    pre_param_.mean.x = kMeanList[0];
    pre_param_.mean.y = kMeanList[1];
    pre_param_.mean.z = kMeanList[2];
    pre_param_.norm_inv.x = kNormList[0];
    pre_param_.norm_inv.y = kNormList[1];
    pre_param_.norm_inv.z = kNormList[2];
    imgpcs_kernel_->convert_normalization_params(pre_param_.mean, pre_param_.norm_inv);
    imgpcs_kernel_->init_mem(pre_param_.src_size.y*pre_param_.src_step, kBatchSize);
    imgpcs_kernel_->set_out_ptr(trt_runner_->GetDevInPtr(sInputNameList[0]));

    LOG_INFO(logger, sIdentifier, "initialization completed");
    return 0;
}

void SixdRepnet360::DrawPoseAxis(cv::Mat& cvmat, const Result& res, const std::vector<Crop>& crop_vec) {
    assert(res.angle_vec.size() == crop_vec.size());
    for (unsigned i = 0; i < res.angle_vec.size(); i++) {
        const auto& pitch = res.angle_vec[i][0];
        const auto& yaw   = 0.0f - res.angle_vec[i][1];
        const auto& roll  = res.angle_vec[i][2];
        const int cx = crop_vec[i].l + static_cast<int>(crop_vec[i].w * 0.5f);
        const int cy = crop_vec[i].t + static_cast<int>(crop_vec[i].h * 0.5f);
        float scale = sqrt((crop_vec[i].w*crop_vec[i].w) + (crop_vec[i].h*crop_vec[i].h))*0.6f;
        // X-Axis pointing to right. drawn in red
        int x1 = static_cast<int>(scale * (cos(yaw) * cos(roll))) + cx;
        int y1 = static_cast<int>(scale * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw))) + cy;

        // Y-Axis | drawn in green
        //        v
        int x2 = static_cast<int>(scale * (-cos(yaw) * sin(roll))) + cx;
        int y2 = static_cast<int>(scale * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll))) + cy;

        // Z-Axis (out of the screen) drawn in blue
        int x3 = static_cast<int>(scale * (sin(yaw))) + cx;
        int y3 = static_cast<int>(scale * (-cos(yaw) * sin(pitch))) + cy;

        cv::line(cvmat, cv::Point(cx, cy), cv::Point(x1, y1), cv::Scalar(0,0,255), 4);
        cv::line(cvmat, cv::Point(cx, cy), cv::Point(x2, y2), cv::Scalar(0,255,0), 4);
        cv::line(cvmat, cv::Point(cx, cy), cv::Point(x3, y3), cv::Scalar(255,0,0), 4);
    }
}

static void RotationMat2EulerAngle(const float* roat_mat_ptr, std::array<float, 3>& eular_angle) {
    float sy = sqrt(roat_mat_ptr[0]*roat_mat_ptr[0]+roat_mat_ptr[3]*roat_mat_ptr[3]);
    float singular = 0.0f;
    if (sy < 1e-6f) {singular = 1.0f;}

    float x = atan2(roat_mat_ptr[7], roat_mat_ptr[8]);
    float y = atan2(0.0f-roat_mat_ptr[6], sy);
    float z = atan2(roat_mat_ptr[3], roat_mat_ptr[0]);
    float xs = atan2(0.0f-roat_mat_ptr[5], roat_mat_ptr[4]);
    float ys = atan2(0.0f-roat_mat_ptr[6], sy);
    float zs = 0.0f;
    eular_angle[0] = x*(1.0f-singular)+xs*singular; // pitch
    eular_angle[1] = y*(1.0f-singular)+ys*singular; // yaw
    eular_angle[2] = z*(1.0f-singular)+zs*singular; // roll
}

static void SetCropAttr(PreParam& param, const Crop& src_crop, const unsigned& img_h, const unsigned& img_w) {
    float src_crop_cx = src_crop.l + src_crop.w * 0.5f;
    float src_crop_cy = src_crop.t + src_crop.w * 0.5f;
    // check at https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/datasets.py#L259
    // head crop would be expanded by a ratio
    float src_crop_ew = src_crop.w * kExpandRatio;
    float src_crop_eh = src_crop.h * kExpandRatio;
    float extended_l = src_crop_cx - src_crop_ew * 0.5f;
    float extended_t = src_crop_cy - src_crop_eh * 0.5f;
    extended_l = extended_l > 0.0f ? extended_l : 0.0f;
    extended_t = extended_t > 0.0f ? extended_t : 0.0f;

    // check at https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/test.py#L125
    // the expanded face crop is first resized to (255,255) then center croped to (224,224)
    // here we find out the range within which src crop's pixels mapped to dst crop, computation is reduced
    param.src_crop.l = static_cast<unsigned>(extended_l + src_crop_ew * 0.0625f);
    param.src_crop.t = static_cast<unsigned>(extended_t + src_crop_eh * 0.0625f);
    param.src_crop.w = static_cast<unsigned>(src_crop_ew * 0.875f);
    param.src_crop.h = static_cast<unsigned>(src_crop_eh * 0.875f);
    if (param.src_crop.l + param.src_crop.w > img_w) {param.src_crop.w = img_w - param.src_crop.l;}
    if (param.src_crop.t + param.src_crop.h > img_h) {param.src_crop.w = img_h - param.src_crop.t;}
    param.dst_crop.l = 0;
    param.dst_crop.t = 0;
    param.dst_crop.w = 224;
    param.dst_crop.h = 224;
    param.scale_inv.x = src_crop_ew / 256.f;
    param.scale_inv.y = src_crop_eh / 256.f;
}

void SixdRepnet360::Process(const cv::Mat& input_cvmat, const std::vector<Crop>& src_crops, Result& result) {
    // 0. input check && calc crops attr
    if (input_cvmat.rows != pre_param_.src_size.y) {LOG_ERROR(logger, sIdentifier, "mat size should be the same as setting's"); exit(1);}
    if (input_cvmat.cols != pre_param_.src_size.x) {LOG_ERROR(logger, sIdentifier, "mat size should be the same as setting's"); exit(1);}
    if (pre_param_.src_step != input_cvmat.step[0] / input_cvmat.step[1]) {LOG_ERROR(logger, sIdentifier, "mat step should be the same as setting's"); exit(1);}
    const unsigned& cur_batch_size = src_crops.size();
    std::vector<PreParam> pre_param_vec(src_crops.size(), pre_param_);
    for (unsigned i = 0; i < src_crops.size(); i++) {
        SetCropAttr(pre_param_vec[i], src_crops[i], input_cvmat.rows, input_cvmat.cols);
    }
    const auto& t_pre_process0 = std::chrono::steady_clock::now();

    // 1. prep-rocess
    const std::string& input_name = sInputNameList[0];
    const std::string& output_name = sOutputNameList[0];
    CHECK_CUDA_FUNC(cudaMemcpyAsync(imgpcs_kernel_->get_dev_in_ptr(0), input_cvmat.data, pre_param_.src_size.y*pre_param_.src_step*3, cudaMemcpyHostToDevice, stream_));
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_pre_process1 = std::chrono::steady_clock::now();
    
    imgpcs_kernel_->preprocess_crops(pre_param_vec, MODEL_IN_RGB,  stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_pre_process2 = std::chrono::steady_clock::now();

    // 2. inference
    trt_runner_->SetCurrentBatchSize(cur_batch_size);
    trt_runner_->InferenceAsync(stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_infer = std::chrono::steady_clock::now();

    trt_runner_->TransOutAsync(stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_post_process0 = std::chrono::steady_clock::now();

    // 3.1 post-process, retrive output rotation mat
    const float* output = (float*)trt_runner_->GetHostOutPtr(output_name);
    result.angle_vec.reserve(cur_batch_size);
    for (unsigned i = 0; i < cur_batch_size; i++) {
        std::array<float,3> eular_angle;
        RotationMat2EulerAngle(output+i*kOutputLen*kOutputChannelNum, eular_angle);
        result.angle_vec.push_back(eular_angle);
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    result.process_time = (t_post_process1-t_pre_process0).count() * 1e-6f;
    char infos[6][kLogInfoSize];
    std::cout << "---------------------\n";
    SET_TIMING_INFO(infos[0], "host_to_device", (t_pre_process1-t_pre_process0).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[1], "pre-process   ", (t_pre_process2-t_pre_process1).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[2], "inference     ", (t_infer-t_pre_process2).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[3], "device_to_host", (t_post_process0-t_infer).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[4], "post-process  ", (t_post_process1-t_post_process0).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[5], "total         ", result.process_time, "ms");
    LOG_INFO(logger, sIdentifier, infos[0]);
    LOG_INFO(logger, sIdentifier, infos[1]);
    LOG_INFO(logger, sIdentifier, infos[2]);
    LOG_INFO(logger, sIdentifier, infos[3]);
    LOG_INFO(logger, sIdentifier, infos[4]);
    LOG_INFO(logger, sIdentifier, infos[5]);
}

void SixdRepnet360::Finalize() {
    imgpcs_kernel_->dest();
    imgpcs_kernel_.reset();
    trt_runner_->Finalize();
    trt_runner_.reset();
    CHECK_CUDA_FUNC(cudaStreamDestroy(stream_));
}