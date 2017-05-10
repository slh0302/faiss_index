find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(CUDA)
find_package(OpenCV REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

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

set(LIB_SSD_LIST ssd.cpp)
add_library(ssd SHARED ${LIB_SSD_LIST})
set(LIB_FEATURE_LIST feature.cpp)
add_library(feature SHARED ${LIB_FEATURE_LIST})


set(SOURCE_FILES IVF.cpp)
add_executable(faissIVF ${SOURCE_FILES})

set(SOURCE_PQ_FILES faissIVFPQ.cpp)
add_executable(faissPQ ${SOURCE_PQ_FILES})

set(SOURCE_PQH_FILES faissIVFPQHam.cpp)
add_executable(faissPQH ${SOURCE_PQH_FILES})

set(SOURCE_GPU_FILES faissGpu_index.cpp)
add_executable(faissGpuPQ ${SOURCE_GPU_FILES})

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
        faissGpuPQ
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