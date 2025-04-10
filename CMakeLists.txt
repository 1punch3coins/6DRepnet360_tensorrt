cmake_minimum_required(VERSION 3.22)
set(PROJECT_NAME head_pose)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
if (MSVC)   # MSVC-specific flags
    set(CMAKE_CXX_FLAGS_RELEASE "/std:c++17 /W3 /O2 /DNDEBUG /wd4996 /wd4244")
    set(CMAKE_CXX_FLAGS_DEBUG "/std:c++17 /Od /Zi /DEBUG")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
else()  # GCC/Clang flags
    set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O0 -g")
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
endif()

set(DEMO0_NAME head_det)
set(DEMO1_NAME head_det_video)
set(DEMO2_NAME head_pose_est)
set(DEMO3_NAME head_pose_est_video)

set(DEVICE_ARCH     "75" CACHE STRING "The device's compute arch to build and run plugin")
set(IS_INPUT_RGB     true CACHE BOOL "The algo's model's input is of rgb or bgr")
set(IS_LOG_PERF      true CACHE BOOL "Whether log model's performance or not")
set(USE_COLOR_LOGGER true CACHE BOOL "Whether use colored logger or not")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(OpenCV_PATH "E:/libs/opencv/build")
    set(CUDA_PATH "E:/libs/Cuda-12.8")
    set(Tensorrt_PATH "E:/libs/TensorRT-10.8.0.43")
    list(APPEND CMAKE_PREFIX_PATH ${OpenCV_PATH} ${CUDA_PATH})
endif()

################ find cuda and tensorrt
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++17,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++17 -O0 -g")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(Tensorrt_INCLUDE_DIRS ${Tensorrt_PATH}/include) # tensorrt does not provide a cmake config file, so manually set lib vars
    set(Tensorrt_LIB_DIRS ${Tensorrt_PATH}/lib)
    link_directories(${Tensorrt_LIB_DIRS})
    set(Tensorrt_LIBS nvinfer_10 nvinfer_plugin_10)
    set(USE_ENQUEUEV3 true)
else()
    set(Tensorrt_LIBS nvinfer nvinfer_plugin cudnn) # tensorrt does not provide a cmake config file, however, it installs the libs under linux system lib paths, so no need to set lib vars manually
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
        if (${TRT_MAJOR_VERSION} GREATER_EQUAL 10)
            set(USE_ENQUEUEV3 true)
        endif()
    else()
        message(STATUS "Version extraction failed.")
    endif()
endif()

################ find opencv
find_package(OpenCV 4.0 REQUIRED)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_videoio)

################ set trtrun_core source files and target source filles
aux_source_directory(src/algos ALGO_SRC_FILES)
if (USE_ENQUEUEV3)
    set(NV_SRC_FILES src/nvprocess/img_precess_k.cu src/nvprocess/img_precess.cpp 
    src/nvprocess/trt_logger.cpp src/nvprocess/trt_runner_v3.cpp)
else()
    set(NV_SRC_FILES src/nvprocess/img_precess_k.cu src/nvprocess/img_precess.cpp 
    src/nvprocess/trt_logger.cpp src/nvprocess/trt_runner_v2.cpp)
endif()

################ add lib trtrun_core
add_library(trtrun_core SHARED ${NV_SRC_FILES})
target_compile_definitions(trtrun_core PUBLIC API_EXPORT)
target_include_directories(trtrun_core PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} src/nvprocess)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_include_directories(trtrun_core PUBLIC ${Tensorrt_INCLUDE_DIRS})
endif()
target_link_libraries(trtrun_core ${CUDA_LIBRARIES} ${Tensorrt_LIBS} ${OpenCV_LIBS})
set_target_properties(trtrun_core PROPERTIES CUDA_ARCHITECTURES ${DEVICE_ARCH})   # make sure half is supported

################ add target demos
message("-- Use enqueuV3: "${USE_ENQUEUEV3})
add_executable(${DEMO0_NAME} ${ALGO_SRC_FILES} src/demos/head_det.cpp)
add_executable(${DEMO1_NAME} ${ALGO_SRC_FILES} src/demos/head_det_video.cpp)
add_executable(${DEMO2_NAME} ${ALGO_SRC_FILES} src/demos/pose_est.cpp)
add_executable(${DEMO3_NAME} ${ALGO_SRC_FILES} src/demos/pose_est_video.cpp)
if(MSVC)
    add_dependencies(${DEMO0_NAME} trtrun_core) # set expicitly the build order
    add_dependencies(${DEMO1_NAME} trtrun_core)
    add_dependencies(${DEMO2_NAME} trtrun_core)
    add_dependencies(${DEMO3_NAME} trtrun_core)
endif()

function(decorator target_name)
    target_compile_definitions(${target_name} PUBLIC USE_ENQUEUEV3=${USE_ENQUEUEV3})
    target_compile_definitions(${target_name} PUBLIC PRINT_TIMING=${IS_LOG_PERF} MODEL_IN_RGB=${IS_INPUT_RGB} USE_COLOR_LOGGER=${USE_COLOR_LOGGER})
    target_include_directories(${target_name} PUBLIC src/algos src/nvprocess)
    target_link_libraries(${target_name} ${OpenCV_LIBS} ${CUDA_LIBRARIES} trtrun_core)
endfunction()

decorator(${DEMO0_NAME})
decorator(${DEMO1_NAME})
decorator(${DEMO2_NAME})
decorator(${DEMO3_NAME})


