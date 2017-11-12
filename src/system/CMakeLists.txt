include_directories(
        ${PROJECT_SOURCE_DIR}/include
        /usr/local/include
        /usr/include
        /usr/local/cuda/include
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/include
        ${PROJECT_SOURCE_DIR}/../
        ${PROJECT_SOURCE_DIR}/../faiss/gpu/
        ${PROJECT_SOURCE_DIR}/../faiss/gpu/utils
        /opt/intel/compilers_and_libraries/linux/mkl/include/
        /usr/local/cuda/include
)
link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/../faiss
        ${PROJECT_SOURCE_DIR}/../faiss/gpu/
        /usr/lib
        /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64
        /usr/local/lib
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib
)


set(SOURCE_UDP_CLIENT client.cpp)
add_executable(UDPClient ${SOURCE_UDP_CLIENT})

set(SOURCE_UDP_SERVER Server.cpp)
add_executable(UDPServer ${SOURCE_UDP_SERVER})

set(SOURCE_MULTI_SERVER MutiUDPServer.cpp)
add_executable(MutiUDPServer ${SOURCE_MULTI_SERVER})

set(SOURCE_VIDEO_SYSTEM VideoRetrievalSystem.cpp)
add_executable(VideoRetrievalSystem ${SOURCE_VIDEO_SYSTEM})

add_library(threadbuf SHARED threadbuf.cpp)

target_link_libraries(
        VideoRetrievalSystem
        gpufaiss
        libfaiss.a
        libbinary.so
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libboost_thread.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libopenblas.so.0
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
        libcudart.so
        libcublas.so
)

target_link_libraries(
        MutiUDPServer
        libretrieval.a
        gpufaiss
        libfaiss.a
        libbinary.so
        libfeature.so
        libssd.so
        libthreadbuf.so
        libboost_system.so
        libboost_filesystem.so
        libboost_thread.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libopenblas.so.0
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
        libcudart.so
        libcublas.so
)

target_link_libraries(
        UDPServer
)


target_link_libraries(
        UDPClient
)