find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        /usr/local/include
        /usr/include
        /home/dell/caffe-ssd/include
        /home/dell/project
        /usr/local/cuda-8.0/include
)
link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        /home/dell/project/faiss
        /usr/lib
        /usr/local/lib
        /home/dell/caffe-ssd/build/lib
)

set(LIB_SSD_LIST ssd.cpp)
add_library(ssd SHARED ${LIB_SSD_LIST})
set(LIB_FEATURE_LIST feature.cpp)
add_library(feature SHARED ${LIB_FEATURE_LIST})


set(SOURCE_FILES main.cpp)
add_executable(faiss_index ${SOURCE_FILES})

target_link_libraries(
        faiss_index
        libfaiss.a
        libopenblas.so.0
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        /home/dell/caffe-ssd/build/lib/libcaffe.so
        libglog.so
        /usr/lib/lapack/liblapack.so.3.0
)