#include <iostream>
#include "yolo11_head.hpp"
#include "sixd_repnet360.hpp"

static void test_6drepnet(const std::string& input_path, const std::string& output_path, const std::string& head_model_path, const std::string& pose_model_path) {
    Yolo11Head head_model;
    SixdRepnet360 head_pose_model;
    cv::Mat ori_img = cv::imread(input_path);
    if (ori_img.empty()) {
        std::cerr << "could not read image from " << input_path << std::endl;
        return;        
    }

    if (head_model.Initialize(head_model_path, ori_img.rows, ori_img.cols)) {
        std::cerr << "Yolo11Head initialization uncompleted" << std::endl;
        return;
    }
    if (head_pose_model.Initialize(pose_model_path, ori_img.rows, ori_img.cols)) {
        std::cerr << "6DRepNet360 initialization uncompleted" << std::endl;
        return;
    }

    Yolo11Head::Result det_res;
    SixdRepnet360::Result pose_res;
    std::vector<Crop> head_crops;
    head_model.Process(ori_img, det_res);
    for (const auto& box: det_res.bbox_vec) {
        head_crops.emplace_back(box.x, box.y, box.w, box.h);
    }
    head_pose_model.Process(ori_img, head_crops, pose_res);

    for (const auto& box: det_res.bbox_vec) {
        cv::putText(ori_img, std::to_string(box.conf).substr(0,4), cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(ori_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
    }
    head_pose_model.DrawPoseAxis(ori_img, pose_res, head_crops);

    cv::imshow("pose_res", ori_img);
    cv::waitKey(0);
    cv::imwrite(output_path, ori_img);
    head_model.Finalize();
    head_pose_model.Finalize();
}

static void help() {
    printf(
        "./head_pose_est [arg0] [arg1] [optional arg2] [optional arg3]\n"
        "arg0: the path of input image\n"
        "arg1: the path of otuput image\n"
        "optional arg2: the path of yolo11 head detection tensorrt model\n"
        "optional arg3: the path of 6DRepNet360 tensorrt model\n"
    );
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        help();
    }
    std::string input_path = std::string(argv[1]);
    std::string output_path = std::string(argv[2]);
    std::string head_model_path = "../model/yolo11n_human_head_1x640x640.plan";
    std::string pose_model_path = "../model/6DRepNet360_nx224x224.plan";
    if (argc == 5) {
        head_model_path = std::string(argv[3]);
        pose_model_path = std::string(argv[4]);
    }

    test_6drepnet(input_path, output_path, head_model_path, pose_model_path);

    return 0;
}
