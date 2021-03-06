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

set(SOURCE_GPU_FILES faissGpu_index.cpp)
add_executable(faissGpuPQ ${SOURCE_GPU_FILES})

set(SOURCE_GPU_IO_IVFPQ GpuIOIVFPQ.cpp)
add_executable(faissGPU_IVFPQ_IO ${SOURCE_GPU_IO_IVFPQ})

set(SOURCE_GPU_IVF faissGPU_IVF.cpp)
add_executable(faissGPU_IVF ${SOURCE_GPU_IVF})

set(SOURCE_GPU_IO_IVF faissGpuIO_IVF.cpp)
add_executable(faissGPU_IO_IVF ${SOURCE_GPU_IO_IVF})


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

target_link_libraries(
        faissGPU_IVFPQ_IO
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
        faissGPU_IVF
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
        faissGPU_IO_IVF
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