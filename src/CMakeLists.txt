find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(CUDA)

SET(OpenCV_DIR /usr/local/share/OpenCV/OpenCVConfig.cmake)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    message("opencv found ${OpenCV_INCLUDE_DIRS} ")
endif()


set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/include
        /usr/include
        /share/local/included
        /usr/local/cuda/include
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/include
        ${PROJECT_SOURCE_DIR}/../
        ${PROJECT_SOURCE_DIR}/../faiss/gpu/
        ${PROJECT_SOURCE_DIR}/../faiss/gpu/utils
        /usr/local/cuda/include
)
link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/../faiss
        ${PROJECT_SOURCE_DIR}/../faiss/gpu/
        /usr/lib
        /share/local/lib
        /usr/local/lib
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib
)

add_subdirectory(util)
add_subdirectory(SQL)
add_subdirectory(gpu)
add_subdirectory(cpu)
add_subdirectory(test)
add_subdirectory(binary)
















