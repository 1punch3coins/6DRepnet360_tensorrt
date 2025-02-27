#include <iostream>
#include <chrono>
#if __GNUC__ > 8
#include <filesystem>
namespace fs = std::filesystem;
#else 
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "yolo11_head.hpp"
#include "sixd_repnet360.hpp"

void test_Yolo11Head(const std::string& input_pwd, const std::string& otuput_pwd, const std::string& model_pwd) {
    Yolo11Head yolo;
    cv::Mat ori_img = cv::imread(input_pwd);
    if (yolo.Initialize(model_pwd)) {
        std::cerr << "Yolo11Head initialization uncompleted" << std::endl;
        return;
    }
    Yolo11Head::Result det_res;
    std::vector<cv::Mat> img_vec;
    img_vec.push_back(ori_img);
    yolo.Process(img_vec, det_res);
    for (const auto& bbox_vec : det_res.batched_bbox_vec) {
        for (const auto& box: bbox_vec) {
            cv::putText(ori_img, std::to_string(box.conf).substr(0,4), cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(ori_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imwrite(otuput_pwd, ori_img);
    yolo.Finalize();
}

void test_6drepnet(const std::string& input_pwd, const std::string& otuput_pwd, const std::string& head_model_pwd, const std::string& pose_model_pwd) {
    Yolo11Head yolo;
    SixdRepnet360 repnet;
    cv::Mat ori_img = cv::imread(input_pwd);
    if (yolo.Initialize(head_model_pwd)) {
        std::cerr << "Yolo11Head initialization uncompleted" << std::endl;
        return;
    }
    if (repnet.Initialize(pose_model_pwd, ori_img.rows, ori_img.cols)) {
        std::cerr << "rep_net6d initialization uncompleted" << std::endl;
        return;
    }

    Yolo11Head::Result det_res;
    SixdRepnet360::Result pose_res;
    std::vector<Crop> head_crops;
    std::vector<cv::Mat> img_vec;
    img_vec.push_back(ori_img);
    yolo.Process(img_vec, det_res);
    for (const auto& bbox_vec : det_res.batched_bbox_vec) {
        for (const auto& box: bbox_vec) {
            head_crops.emplace_back(box.x, box.y, box.w, box.h);
        }
    }
    repnet.Process(ori_img, head_crops, pose_res);

    for (const auto& bbox_vec : det_res.batched_bbox_vec) {
        for (const auto& box: bbox_vec) {
            cv::putText(ori_img, std::to_string(box.conf).substr(0,4), cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(ori_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        }
    }
    repnet.DrawPoseAxis(ori_img, pose_res, head_crops);
    cv::imwrite(otuput_pwd, ori_img);
    yolo.Finalize();
    repnet.Finalize();
}

// void test_6drepnet(const std::string& input_pwd, const std::string& head_model_pwd, const std::string& pose_model_pwd) {
//     Yolo11Head yolo;
//     SixdRepnet360 repnet;
//     cv::VideoCapture cap(input_pwd);
    
//     if (yolo.Initialize(head_model_pwd)) {
//         std::cerr << "Yolo11Head initialization uncompleted" << std::endl;
//         return 1;
//     }
//     if (repnet.Initialize(pose_model_pwd, cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH))) {
//         std::cerr << "rep_net6d initialization uncompleted" << std::endl;
//         return 1;
//     }

//     cv::Mat frame;
//     while (cap.read(frame)) {
//         Yolo11Head::Result det_res;
//         SixdRepnet360::Result pose_res;
//         std::vector<Crop> head_crops;
//         yolo.Process(frame, det_res);
//         head_crops.reserve(det_res.bbox_vec.size());
//         for (const auto& box: det_res.bbox_vec) {
//             head_crops.emplace_back(box.x, box.y, box.w, box.h);
//         }
//         repnet.Process(frame, head_crops, pose_res);

//         for (const auto& box: det_res.bbox_vec) {
//             cv::putText(frame, std::to_string(box.conf).substr(0,4), cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
//             cv::rectangle(frame, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
//         }
//         repnet.DrawPoseAxis(frame, pose_res, head_crops);
//         cv::imshow("result", frame);
//         if (cv::waitKey(30) == 27) {
//             break;
//         }
//     }
//     yolo.Finalize();
//     repnet.Finalize();
// }

static void get_in_img_paths(const char *folder_dir, std::vector<std::string>& file_paths, const char *suffix = ".jpg") {
    if (fs::is_directory(folder_dir)) {
        for (const auto& entry: fs::directory_iterator(folder_dir)) {
            if (entry.path().extension() == suffix) {
                file_paths.push_back(entry.path());
            }
        }
    } else {
        printf("directory %s is not accessible\n", folder_dir);
        exit(EXIT_FAILURE);
    }
}

static void set_out_img_paths(const char *folder_dir, const std::vector<std::string>& in_file_paths, std::vector<std::string>& out_file_paths) {
    if (fs::is_directory(folder_dir)) {
        for (const auto& s: in_file_paths) {
            auto out_file_path = fs::path(folder_dir);
            auto in_file_path = fs::path(s);
            out_file_path.append((in_file_path.stem().concat("_res")+=in_file_path.extension()).c_str());
            out_file_paths.push_back(out_file_path);
        }
    } else {
        printf("directory %s is not accessible\n", folder_dir);
        exit(EXIT_FAILURE);
    }
}

int main(void) {   
    // test_Yolo11Head("../asset/input/faces6.jpg", "../asset/output/faces6.jpg", "../model/Yolo11Headn_human_head_1x640x640.plan");
    test_6drepnet("../asset/input/faces4.jpg", "../asset/output/faces4.jpg", "../model/yolo11n_human_head_1x640x640.plan", "../model/6DRepNet360_nx224x224.plan");
    // test_6drepnet("../asset/input/dancing.mp4", "../model/Yolo11Headn_1x640x640.plan", "../model/6DRepNet_nx224x224.plan");

    return 0;
}

// int main(int argc, char* argv[]) {
//     if (argc != 4) {
//         help();
//     }
//     const char* model_pwd = argv[1];
//     const char* in_dir    = argv[2];
//     const char* out_dir   = argv[3];
//     std::vector<std::string> input_pwds, output_pwds;
//     get_in_img_paths(in_dir, input_pwds);
//     set_out_img_paths(out_dir, input_pwds, output_pwds);
//     test_yolo11(input_pwds, output_pwds, model_pwd);
//     return 0;
// }