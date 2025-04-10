# 6DRepNet360_Tensorrt
This project is for head pose estimation. It uses a tensorrt-based implemntation of [6DRepNet360](https://github.com/thohemp/6DRepNet360/blob/master/README.MD) in c++ environment.
<p align="center">
  <img src="asset/boy_in_a_mirror.gif" width="640" />
</p>

## Prepare
Make sure msvc, cmake, opencv(>=4.0), cuda and tenssorrt(>=10.0) are properly installed under windows os.

Clone the repo.
```bash
git clone https://github.com/1punch3coins/6DRepnet360_tensorrt.git
cd 6DRepNet360-Tensorrt
```
The repo uploads a trained yolo11n head detection onnx file and a 6DRepNet360 head pose estimation onnx file under model folder. You could directly export them into trt engine files using the following commands:
```bash
path/to/tensorrt/bin/trtexec.exe --onnx=model/yolo11n_human_head_1x640x640.onnx --saveEngine=model/yolo11n_human_head_1x640x640.plan --fp16 --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > model/yolo11n_human_head_trt_build.log
path/to/tensorrt/bin/trtexec.exe --onnx=model/6DRepNet360_nx224x224.onnx --saveEngine=model/6DRepNet360_nx224x224.plan --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:16x3x224x224 --fp16 --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > model/6DRepNet_trt_build.log
```
After seconds to minutes waiting, the two exported plan files and logs would be under the model folder too.
## Build
Before compilation, set your opencv, cuda and tensorrt installation paths in repo's CMakeLists.txt [here](CMakeLists.txt#L27).

Check your gpu's compute capacity [here](https://developer.nvidia.com/cuda-gpus), and build the codes as following.
```bash
mkdir build; cd build
cmake .. -DDEVICE_ARCH=/your/device/capacity
cmake --build . --config Release -j
```
## Usage
Append your cuda, tensorrt, and opencv libs' paths to system environment variables temporarily under shell:
```bash
$env:PAHT+="paht_to_cuda_lib;path_to_tensorrt_lib;path_to_opencv_lib;"
```
To run a 6DRepNet360 head pose estimation with a single image, run:
```bash
./Release/head_pose_est.exe /path/to/input_image /path/to/output_image
```
To run a 6DRepNet360 head pose estimation with a video, run:
```bash
./Release/head_pose_est_video.exe /path/to/input_video /path/to/output_video
```
## License
The repo is under MIT license.
## References
🔗 [6DRepNet360 official repo](https://github.com/thohemp/6DRepNet360/blob/master/README.MD)
🔗 [Yolo11 official repo](https://github.com/ultralytics/ultralytics.git)
🔗 [An awesome 6DRepNet360 implementation in onnx-runtime python](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/423_6DRepNet360)