set(SOURCE_OLD_SERVER OldSyetem.cpp)
add_executable(oldSystem ${SOURCE_OLD_SERVER})

set(EXTRACT_ATTR_SERVER extractColorType.cpp)
add_executable(extractColorType ${EXTRACT_ATTR_SERVER})

target_link_libraries(
        oldSystem
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
        extractColorType
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