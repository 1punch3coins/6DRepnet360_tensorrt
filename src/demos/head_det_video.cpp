#include <iostream>
#include "yolo11_head.hpp"
#include "sixd_repnet360.hpp"

static void test_6drepnet(const std::string& input_path, const std::string& output_path, const std::string& head_model_path) {
    Yolo11Head head_model;
    cv::VideoCapture cap(input_path);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

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

    cv::Mat frame;
    while (cap.read(frame)) {
        Yolo11Head::Result det_res;
        head_model.Process(frame, det_res);
        for (const auto& box: det_res.bbox_vec) {
            cv::putText(frame, std::to_string(box.conf).substr(0,4), cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(frame, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("result", frame);
        auto t0 = std::chrono::steady_clock::now();
        writer.write(frame);
        auto t1 = std::chrono::steady_clock::now();
        float wait_time = (1000.0f/fps-(det_res.process_time+(t1-t0).count()*1e-6f));
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
}

static void help() {
    printf(
        "./head_det_video [arg0] [arg1] [optional arg2]\n"
        "arg0: the path of input video\n"
        "arg1: the path of otuput video\n"
        "optional arg2: the path of yolo11 head detection tensorrt model\n"
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
    if (argc == 4) {
        head_model_path = std::string(argv[3]);
    }

    test_6drepnet(input_path, output_path, head_model_path);

    return 0;
}


