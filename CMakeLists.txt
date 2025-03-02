cmake_minimum_required(VERSION 3.22)
set(PROJECT_NAME head_pose)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g")

set(DEMO0_NAME head_det)
set(DEMO1_NAME head_det_video)
set(DEMO2_NAME head_pose_est)
set(DEMO3_NAME head_pose_est_video)

set(DEVICE_ARCH     "75" CACHE STRING "The device's compute arch to build and run plugin")
set(IS_INPUT_RGB     true CACHE BOOL "The algo's model's input is of rgb or bgr")
set(IS_LOG_PERF      true CACHE BOOL "Whether log model's performance or not")
set(USE_COLOR_LOGGER true CACHE BOOL "Whether use colored logger or not")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++17,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++17 -O0 -g")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
set(CUDA_LIBS ${CUDA_LIBRARIES} nvinfer nvonnxparser nvinfer_plugin cudnn)
################ find libnvinfer
find_library(NVINFER_LIB_LOC nvinfer)
execute_process(
    COMMAND readlink ${NVINFER_LIB_LOC}
    OUTPUT_VARIABLE NVINFER_LIB_NAME
    ERROR_VARIABLE LINK_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
################ check tensorrt version
string(REGEX MATCH "libnvinfer\.so\.([0-9][0-9])\.([0-9])\.*" REGEX_RES ${NVINFER_LIB_NAME})
set(USE_ENQUEUEV3 false)
if(REGEX_RES)
    set(TRT_MAJOR_VERSION ${CMAKE_MATCH_1}) 
    set(TRT_MINOR_VERSION ${CMAKE_MATCH_2})
    message("-- Found tensorrt_version: "${TRT_MAJOR_VERSION}.${TRT_MINOR_VERSION})
    if (${TRT_MAJOR_VERSION} GREATER_EQUAL 9 OR (${TRT_MAJOR_VERSION} EQUAL 8 AND ${TRT_MINOR_VERSION} GREATER_EQUAL 5))
        set(USE_ENQUEUEV3 true)
    endif()
else()
    message(STATUS "Version extraction failed.")
endif()

find_package(OpenCV 4.0 REQUIRED)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_videoio)

aux_source_directory(src/algos ALGO_SRC_FILES)
if (USE_ENQUEUEV3)
    set(NV_SRC_FILES src/nvprocess/img_precess_k.cu src/nvprocess/img_precess.cpp 
        src/nvprocess/trt_logger.cpp src/nvprocess/trt_runner_v3.cpp)
else()
    set(NV_SRC_FILES src/nvprocess/img_precess_k.cu src/nvprocess/img_precess.cpp 
        src/nvprocess/trt_logger.cpp src/nvprocess/trt_runner_v2.cpp)
endif()

####################### trtrun_core.so ########################
add_library(trtrun_core SHARED ${NV_SRC_FILES})
target_compile_definitions(trtrun_core PUBLIC)
target_include_directories(trtrun_core PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} src/nvprocess)
target_link_libraries(trtrun_core ${CUDA_LIBS} ${OpenCV_LIBS} )
set_target_properties(trtrun_core PROPERTIES CUDA_ARCHITECTURES ${DEVICE_ARCH})   # make sure half is supported

######################## target demos #########################
message("-- Use enqueuV3: "${USE_ENQUEUEV3})
add_executable(${DEMO0_NAME} ${ALGO_SRC_FILES} src/demos/head_det.cpp)
add_executable(${DEMO1_NAME} ${ALGO_SRC_FILES} src/demos/head_det_video.cpp)
add_executable(${DEMO2_NAME} ${ALGO_SRC_FILES} src/demos/pose_est.cpp)
add_executable(${DEMO3_NAME} ${ALGO_SRC_FILES} src/demos/pose_est_video.cpp)

function(decorator target_name)
    target_compile_definitions(${target_name} PUBLIC USE_ENQUEUEV3)
    target_compile_definitions(${target_name} PUBLIC PRINT_TIMING=${IS_LOG_PERF} MODEL_IN_RGB=${IS_INPUT_RGB} USE_COLOR_LOGGER=${USE_COLOR_LOGGER})
    target_include_directories(${target_name} PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} src/nvprocess src/algos)
    target_link_libraries(${target_name} ${OpenCV_LIBS} trtrun_core)
endfunction()

decorator(${DEMO0_NAME})
decorator(${DEMO1_NAME})
decorator(${DEMO2_NAME})
decorator(${DEMO3_NAME})


