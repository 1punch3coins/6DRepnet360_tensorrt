#include <iostream>
#include "yolo11_head.hpp"

static void test_yolo11_head(const std::string& input_path, const std::string& output_path, const std::string& model_path) {
    Yolo11Head head_model;
    cv::Mat ori_img = cv::imread(input_path);
    if (head_model.Initialize(model_path, ori_img.rows, ori_img.cols)) {
        std::cerr << "Yolo11Head initialization uncompleted" << std::endl;
        return;
    }
    Yolo11Head::Result det_res;
    head_model.Process(ori_img, det_res);
    for (const auto& box: det_res.bbox_vec) {
        cv::putText(ori_img, std::to_string(box.conf).substr(0,4), cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(ori_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("head_res", ori_img);
    cv::waitKey(0);

    cv::imwrite(output_path, ori_img);
    head_model.Finalize();
}

static void help() {
    printf(
        "./head_det [arg0] [arg1] [optional arg2]\n"
        "arg0: the path of input image\n"
        "arg1: the path of otuput image\n"
        "optional arg2: the path of yolo11 head detection tensorrt model\n"
    );
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        help();
    }
    std::string input_path = std::string(argv[1]);
    std::string output_path = std::string(argv[2]);
    std::string model_path = "../model/yolo11n_human_head_1x640x640.plan";
    if (argc == 4) {
        model_path = std::string(argv[3]);
    }

    test_yolo11_head(input_path, output_path, model_path);

    return 0;
}
