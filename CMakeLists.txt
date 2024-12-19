cmake_minimum_required(VERSION 3.10)

project(rkface)

set(ARM_CROSS_COMPILE_TOOLCHAIN /opt/toolchain/arm-rockchip830-linux-uclibcgnueabihf)
set(CMAKE_C_COMPILER ${ARM_CROSS_COMPILE_TOOLCHAIN}/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc)
set(CMAKE_CXX_COMPILER ${ARM_CROSS_COMPILE_TOOLCHAIN}/bin/arm-rockchip830-linux-uclibcgnueabihf-g++)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/rkaiq
    ${PROJECT_SOURCE_DIR}/include/rkaiq/common
    ${PROJECT_SOURCE_DIR}/include/rkaiq/uAPI2
    ${PROJECT_SOURCE_DIR}/include/rkaiq/xcore
    ${PROJECT_SOURCE_DIR}/include/rkaiq/algos
    ${PROJECT_SOURCE_DIR}/include/rkaiq/iq_parser
    ${PROJECT_SOURCE_DIR}/include/rkaiq/iq_parser_v2
    ${PROJECT_SOURCE_DIR}/include/rknn
    ${PROJECT_SOURCE_DIR}/include/opencv4
)

link_directories(
    ${PROJECT_SOURCE_DIR}/lib   
)

add_executable(${PROJECT_NAME}
    src/main.cc
    src/luckfox_mpi.cc
    src/retinaface.cc
    src/facenet.cc
    src/alignment.h
)

target_link_libraries(${PROJECT_NAME} 
    rockit rockchip_mpp rtsp rkaiq sample_comm rknnmrt rockiva rga
    opencv_imgproc opencv_features2d opencv_highgui opencv_photo opencv_video opencv_core
)