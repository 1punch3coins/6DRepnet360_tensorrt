#include <iostream>
#include <fstream>
#include <chrono>
#include <array>
#include <algorithm>
#include "algo_logger.hpp"
#include "yolo11_head.hpp"

// model's input's and output's meta
using Tt = NetMeta::TensorType;
static constexpr int32_t kInputTensorNum = 1;
static constexpr int32_t kOutputTensorNum = 4;
static constexpr int32_t kBatchSize = 1;
static constexpr std::array<const char*, kInputTensorNum> sInputNameList = {"input"};
static constexpr std::array<const bool, kInputTensorNum> iInputNchwList = {true};
static constexpr std::array<const Tt, kInputTensorNum> kInputTypeList = {Tt::kTypeFloat32};
static constexpr std::array<const char*, kOutputTensorNum> sOutputNameList = {"num", "boxes", "classes", "scores"};
static constexpr std::array<const bool, kOutputTensorNum> iOutputNlcList = {true, true, true, true};
static constexpr std::array<const Tt, kOutputTensorNum> kOutputTypeList = {Tt::kTypeInt32, Tt::kTypeFloat32, Tt::kTypeInt32, Tt::kTypeFloat32};
// used for pre_process
static constexpr float kMeanList[] = {0.0f, 0.0f, 0.0f};
static constexpr float kNormList[] = {1.0f, 1.0f, 1.0f};
static constexpr CropStyle kStyle = CropStyle::SrcCropAll_DstEmbeddCnt;
// used for post_process and net output meta check
static constexpr std::array<int32_t, kOutputTensorNum> kOutputChannelList = {1, 4, 1, 1};
static constexpr int32_t kOutputLen = 100;
static constexpr int32_t kClassNum = 80;
static constexpr auto& kBoxNumChannelNum = kOutputChannelList[0];
static constexpr auto& kOutputBoxChannelNum = kOutputChannelList[1];
static constexpr auto& kOutputConfChannelNum = kOutputChannelList[2];
static constexpr auto& kOutputIdChannelNum = kOutputChannelList[3];
// used for logger
static constexpr const char* sIdentifier = {"Yolo11Head"};
static constexpr int32_t kLogInfoSize = 100;
static auto logger = algoLogger::logger;

int32_t Yolo11Head::Initialize(const std::string& model, const int& img_h, const int& img_w) {
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
    if (trt_runner_->InitBatchSize(false, kBatchSize)) {
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
    for (unsigned i = 0; i < sOutputNameList.size(); i++) {
        if (trt_runner_->GetOutChannelNum(sOutputNameList[i]) != kOutputChannelList[i]) {
            trt_runner_.reset();
            LOG_ERROR_RETURN(logger, sIdentifier, "output channel size mismatched");
        }
    }
    for (unsigned i = 1; i < sOutputNameList.size(); i++) {
        if (trt_runner_->GetOutLen(sOutputNameList[i]) != kOutputLen) {
            trt_runner_.reset();
            LOG_ERROR_RETURN(logger, sIdentifier, "output len mismatched");
        }
    }
    LOG_INFO(logger, sIdentifier, "output meta check completed");

    // set image preprocess kernel
    const int32_t net_in_h = trt_runner_->GetInHeight(sInputNameList[0]);
    const int32_t net_in_w = trt_runner_->GetInWidth(sInputNameList[0]);
    imgpcs_kernel_ = std::make_unique<ImgPrecess>();
    pre_param_.src_size.x = img_w;     // input image's size
    pre_param_.src_size.y = img_h;
    pre_param_.dst_size.x = net_in_w;  // output mat's size
    pre_param_.dst_size.y = net_in_h;
    pre_param_.src_step   = img_w;     // input mat's step(it should be each row's byte size; here for kernel's simplicity, it is each row's element num)
    pre_param_.dst_step   = net_in_w;  // output mat's step(it should be each row's byte size; here for kernel's simplicity, it is each row's element num)
    pre_param_.mean.x = kMeanList[0];
    pre_param_.mean.y = kMeanList[1];
    pre_param_.mean.z = kMeanList[2];
    pre_param_.norm_inv.x = kNormList[0];
    pre_param_.norm_inv.y = kNormList[1];
    pre_param_.norm_inv.z = kNormList[2];
    imgpcs_kernel_->convert_normalization_params(pre_param_.mean, pre_param_.norm_inv);
    imgpcs_kernel_->init_mem(pre_param_.src_size.y*pre_param_.src_step, kBatchSize);
    imgpcs_kernel_->set_out_ptr(trt_runner_->GetDevInPtr(sInputNameList[0]));
    imgpcs_kernel_->set_crop_attr(pre_param_, pre_param_.src_size.x, pre_param_.src_size.y, pre_param_.dst_size.x, pre_param_.dst_size.y, kStyle);

    // create cuda stream
    CHECK_CUDA_FUNC(cudaStreamCreate(&stream_));
    LOG_INFO(logger, sIdentifier, "initialization completed");
    return 0;
}

void Yolo11Head::Process(const cv::Mat& input_cvmat, Result& result) {
    // 0. input check
    if (input_cvmat.rows != pre_param_.src_size.y) {LOG_ERROR(logger, sIdentifier, "mat size should be the same as setting's"); exit(1);}
    if (input_cvmat.cols != pre_param_.src_size.x) {LOG_ERROR(logger, sIdentifier, "mat size should be the same as setting's"); exit(1);}
    if (pre_param_.src_step != input_cvmat.step[0] / input_cvmat.step[1]) {LOG_ERROR(logger, sIdentifier, "mat step should be the same as setting's"); exit(1);}
    const auto& t_pre_process0 = std::chrono::steady_clock::now();

    // 1. prep-rocess
    const std::string& input_name = sInputNameList[0];
    const std::string& output_name = sOutputNameList[0];
    CHECK_CUDA_FUNC(cudaMemcpyAsync(imgpcs_kernel_->get_dev_in_ptr(0), input_cvmat.data, pre_param_.src_size.y*pre_param_.src_step*3, cudaMemcpyHostToDevice, stream_));
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_pre_process1 = std::chrono::steady_clock::now();
    
    std::vector<PreParam> pre_param_vec(1, pre_param_);
    imgpcs_kernel_->preprocess_cvmats(pre_param_vec, MODEL_IN_RGB, stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_pre_process2 = std::chrono::steady_clock::now();

    // 2. inference
    trt_runner_->SetCurrentBatchSize(1);
    trt_runner_->InferenceAsync(stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_infer = std::chrono::steady_clock::now();
    
    trt_runner_->TransOutAsync(stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_post_process0 = std::chrono::steady_clock::now();

    // 3. post-process, retrive output; scale and offset bboxes to input img
    const unsigned* dets_num_ptr   = (unsigned*)trt_runner_->GetHostOutPtr(sOutputNameList[0]);
    const float*    boxes_ptr      = (float*)trt_runner_->GetHostOutPtr(sOutputNameList[1]);
    const int*      cls_ids_ptr    = (int*)trt_runner_->GetHostOutPtr(sOutputNameList[2]);
    const float*    cls_scores_ptr = (float*)trt_runner_->GetHostOutPtr(sOutputNameList[3]);

    unsigned dets_num = *dets_num_ptr;
    std::vector<Bbox2D>& bbox_vec = result.bbox_vec;
    bbox_vec.reserve(dets_num);
    const auto& src_crop = pre_param_.src_crop;
    const auto& dst_crop = pre_param_.dst_crop;
    const auto& scale_w  = pre_param_.scale_inv.x;
    const auto& scale_h  = pre_param_.scale_inv.y;
    for (unsigned i = 0; i < dets_num; i++) {
        int cls_id = cls_ids_ptr[i];
        float cls_conf = cls_scores_ptr[i];
        unsigned box_index = i*kOutputBoxChannelNum;
        int x0 = static_cast<int>((boxes_ptr[box_index + 0] - dst_crop.l) * scale_w) + src_crop.l;
        int y0 = static_cast<int>((boxes_ptr[box_index + 1] - dst_crop.t) * scale_h) + src_crop.t;
        int x1 = static_cast<int>((boxes_ptr[box_index + 2] - dst_crop.l) * scale_w) + src_crop.l;
        int y1 = static_cast<int>((boxes_ptr[box_index + 3] - dst_crop.t) * scale_h) + src_crop.t;
        int w = static_cast<int>(x1 - x0);
        int h = static_cast<int>(y1 - y0);
        std::string cls_name = "head";
        bbox_vec.emplace_back(cls_name, cls_id, cls_conf, x0, y0, w, h);
    }

    const auto& t_post_process1 = std::chrono::steady_clock::now();
    result.process_time = 1.0f * (t_post_process1 - t_pre_process0).count() * 1e-6f;
#ifdef PRINT_TIMING
    std::cout << "---------------------\n";
    char infos[6][kLogInfoSize];
    SET_TIMING_INFO(infos[0], "host_to_device", (t_pre_process1-t_pre_process0).count()*1e-6, "ms");
    SET_TIMING_INFO(infos[1], "pre-process   ", (t_pre_process2-t_pre_process1).count()*1e-6, "ms");
    SET_TIMING_INFO(infos[2], "inference     ", (t_infer-t_pre_process2).count()*1e-6, "ms");
    SET_TIMING_INFO(infos[3], "device_to_host", (t_post_process0-t_infer).count()*1e-6, "ms");
    SET_TIMING_INFO(infos[4], "post-process  ", (t_post_process1-t_post_process0).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[5], "total         ", result.process_time, "ms");
    LOG_INFO(logger, sIdentifier, infos[0]);
    LOG_INFO(logger, sIdentifier, infos[1]);
    LOG_INFO(logger, sIdentifier, infos[2]);
    LOG_INFO(logger, sIdentifier, infos[3]);
    LOG_INFO(logger, sIdentifier, infos[4]);
    LOG_INFO(logger, sIdentifier, infos[5]);
#endif
}

void Yolo11Head::Finalize() {
    imgpcs_kernel_->dest();
    imgpcs_kernel_.reset();
    trt_runner_->Finalize();
    trt_runner_.reset();
    CHECK_CUDA_FUNC(cudaStreamDestroy(stream_));
}