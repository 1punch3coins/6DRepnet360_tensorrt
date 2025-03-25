#include <iostream>
#include "yolo11_head.hpp"
#include "sixd_repnet360.hpp"

static void test_6drepnet(const std::string& input_path, const std::string& output_path, const std::string& head_model_path, const std::string& pose_model_path) {
    Yolo11Head head_model;
    SixdRepnet360 head_pose_model;
    cv::VideoCapture cap(input_path);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    
    if (!cap.isOpened()) {
        std::cerr << "could not find or open the video at " << input_path << std::endl;
        return;
    }
    if (!writer.isOpened()) {
        std::cerr << "could not open the video writer at " << output_path << std::endl;
        return;
    }

    if (head_model.Initialize(head_model_path, frame_height, frame_width)) {
        std::cerr << "Yolo11Head initialization uncompleted" << std::endl;
        return;
    }
    if (head_pose_model.Initialize(pose_model_path, frame_height, frame_width)) {
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
        if (head_crops.size() > 0) {
            head_pose_model.Process(frame, head_crops, pose_res);
            for (const auto& box: det_res.bbox_vec) {
                cv::putText(frame, std::to_string(box.conf).substr(0,4), cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
                cv::rectangle(frame, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
            }
            head_pose_model.DrawPoseAxis(frame, pose_res, head_crops);
        }

        cv::imshow("result", frame);
        auto t0 = std::chrono::steady_clock::now();
        writer.write(frame);
        auto t1 = std::chrono::steady_clock::now();
        float wait_time = (1000.0f/fps-(det_res.process_time+pose_res.process_time+(t1-t0).count()*1e-6f));
        wait_time = wait_time < 0.0f ? 1.0f : wait_time;
        if (cv::waitKey(wait_time) == 27) {
            cap.release();
            writer.release();
            break;
        }
    }
    cap.release();
    writer.release();
    head_model.Finalize();
    head_pose_model.Finalize();
}

static void help() {
    printf(
        "./head_pose_est_video [arg0] [arg1] [optional arg2] [optional arg3]\n"
        "arg0: the path of input video\n"
        "arg1: the path of otuput video\n"
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


