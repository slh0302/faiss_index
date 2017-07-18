set(SOURCE_FILES IVF.cpp)
add_executable(faissIVF ${SOURCE_FILES})

set(SOURCE_PQ_FILES faissIVFPQ.cpp)
add_executable(faissPQ ${SOURCE_PQ_FILES})

set(SOURCE_PQH_FILES faissIVFPQHam.cpp)
add_executable(faissPQH ${SOURCE_PQH_FILES})

set(SOURCE_IO_IVFPQ ioIVFPQ.cpp)
add_executable(faissIO_IVFPQ ${SOURCE_IO_IVFPQ})

set(SOURCE_FALT_FILES faissFlat.cpp)
add_executable(faissFlat ${SOURCE_FALT_FILES})

target_link_libraries(
        faissPQ
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
        faissPQH
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
        faissIO_IVFPQ
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
        faissIVF
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
        faissFlat
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
