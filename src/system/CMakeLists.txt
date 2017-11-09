set(SOURCE_UDP_CLIENT client.cpp)
add_executable(UDPClient ${SOURCE_UDP_CLIENT})

set(SOURCE_UDP_SERVER Server.cpp)
add_executable(UDPServer ${SOURCE_UDP_SERVER})

target_link_libraries(
        UDPClient
        gpufaiss
        faiss
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
        UDPServer
        gpufaiss
        faiss
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