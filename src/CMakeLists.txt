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

add_subdirectory(util)
add_subdirectory(gpu)
add_subdirectory(cpu)
add_subdirectory(test)
















