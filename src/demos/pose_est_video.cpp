#include <iostream>
#include "yolo11_head.hpp"
#include "sixd_repnet360.hpp"

static void test_6drepnet(const std::string& input_path, const std::string& output_path, const std::string& head_model_path, const std::string& pose_model_path) {
    Yolo11Head head_model;
    SixdRepnet360 head_pose_model;
    cv::VideoCapture cap(input_path);
    
    if (head_model.Initialize(head_model_path, cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH))) {
        std::cerr << "Yolo11Head initialization uncompleted" << std::endl;
        return;
    }
    if (head_pose_model.Initialize(pose_model_path, cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH))) {
        std::cerr << "6DRepNet360 initialization uncompleted" << std::endl;
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        Yolo11Head::Result det_res;
        SixdRepnet360::Result pose_res;
        std::vector<Crop> head_crops;
        head_model.Process(frame, det_res);
        head_crops.reserve(det_res.bbox_vec.size());
        for (const auto& box: det_res.bbox_vec) {
            head_crops.emplace_back(box.x, box.y, box.w, box.h);
        }
        head_pose_model.Process(frame, head_crops, pose_res);

        for (const auto& box: det_res.bbox_vec) {
            cv::putText(frame, std::to_string(box.conf).substr(0,4), cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(frame, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        }
        head_pose_model.DrawPoseAxis(frame, pose_res, head_crops);
        cv::imshow("result", frame);
        if (cv::waitKey(30) == 27) {
            break;
        }
    }
    head_model.Finalize();
    head_pose_model.Finalize();
}

static void help() {

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


