set(SOURCE_TESTIVF_FILES tests_imi.cpp)
add_executable(tests_imi ${SOURCE_TESTIVF_FILES})

set(SOURCE_TESTIVFPQ_FILES test_ivfpg_demo.cpp)
add_executable(test_ivfpq ${SOURCE_TESTIVFPQ_FILES})

set(SOURCE_TEST_CAR_FEATURE test_feature_car.cpp)
add_executable(test_feature_car ${SOURCE_TEST_CAR_FEATURE})

set(SOURCE_TEST_MEM_FEATURE test_mem_feature.cpp)
add_executable(test_mem_feature ${SOURCE_TEST_MEM_FEATURE})


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
        test_feature_car
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
        test_mem_feature
        gpufaiss
        faiss
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
        libglog.so
        libcudart.so
        libcublas.so
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
