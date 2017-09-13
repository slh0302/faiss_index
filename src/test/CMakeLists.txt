set(SOURCE_TESTIVF_FILES tests_imi.cpp)
add_executable(tests_imi ${SOURCE_TESTIVF_FILES})

set(SOURCE_TESTIVFPQ_FILES test_ivfpg_demo.cpp)
add_executable(test_ivfpq ${SOURCE_TESTIVFPQ_FILES})



target_link_libraries(
        tests_imi
        libfaiss.a
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libopenblas.so.0
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
)

target_link_libraries(
        test_ivfpq
        libfaiss.a
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libopenblas.so.0
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
)
