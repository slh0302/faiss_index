
set(LIB_SSD_LIST ssd.cpp)
add_library(ssd SHARED ${LIB_SSD_LIST})

set(LIB_FEATURE_LIST feature.cpp)
add_library(feature SHARED ${LIB_FEATURE_LIST})


set(SOURCE_FEATURE test.cpp)
add_executable(extract_feature ${SOURCE_FEATURE})

set(SOURCE_RUN_SYSTEM faissSystem.cpp)
add_executable(faissSystem ${SOURCE_RUN_SYSTEM})

set(SOURCE_CAR_SYSTEM faissSystemCar.cpp)
add_executable(faissSystemCar ${SOURCE_CAR_SYSTEM})

set(SOURCE_CAR_IVF_SYSTEM faissSystemCarIVF.cpp)
add_executable(faissSystemCarIVF ${SOURCE_CAR_IVF_SYSTEM})

target_link_libraries(
        extract_feature
        libfaiss.a
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libmkl_intel_ilp64.so
        libmkl_core.so
        libmkl_gnu_thread.so
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
)

target_link_libraries(
        faissSystem
        gpufaiss
        faiss
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libmkl_intel_ilp64.so
        libmkl_core.so
        libmkl_gnu_thread.so
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
        libcudart.so
        libcublas.so
)

target_link_libraries(
        faissSystemCar
        gpufaiss
        faiss
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libmkl_intel_ilp64.so
        libmkl_core.so
        libmkl_gnu_thread.so
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
        libcudart.so
        libcublas.so
)

target_link_libraries(
        faissSystemCarIVF
        gpufaiss
        faiss
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libmkl_intel_ilp64.so
        libmkl_core.so
        libmkl_gnu_thread.so
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
        libcudart.so
        libcublas.so
)


