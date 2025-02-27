#!bin/bash

# build yolo11_haed model
# /usr/src/tensorrt/bin/trtexec --onnx=model/yolo11n_human_head_1x640x640.onnx --saveEngine="model/yolo11n_human_head_1x640x640.plan" --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > model/yolo11n_human_head_trt_build.log 2>&1

# build 6DRepNet360 head pose model, max available heads num is 16
/usr/src/tensorrt/bin/trtexec --onnx="model/6DRepNet360_nx224x224.onnx" --saveEngine="model/6DRepNet360_nx224x224.plan" --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:16x3x224x224 --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > model/6DRepNet_trt_build_f32.log 2>&1