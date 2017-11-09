set(SOURCE_TEST_RECALL TestRecall.cpp)
add_executable(TestRecall ${SOURCE_TEST_RECALL})

set(SOURCE_TEST_MAP TestmAP.cpp)
add_executable(TestmAP ${SOURCE_TEST_MAP})

target_link_libraries(
        TestmAP
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
        TestRecall
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