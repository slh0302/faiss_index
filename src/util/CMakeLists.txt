
set(LIB_SSD_LIST ssd.cpp)
add_library(ssd SHARED ${LIB_SSD_LIST})

set(LIB_FEATURE_LIST feature.cpp)
add_library(feature SHARED ${LIB_FEATURE_LIST})

set(LIB_RETRIEVAL Retrieval.cpp)
add_library(retrieval STATIC ${LIB_RETRIEVAL})

set(SOURCE_FEATURE test.cpp)
add_executable(extract_feature ${SOURCE_FEATURE})

set(TEST_PIC_CPP test_pic.cpp)
add_executable(test_pic ${TEST_PIC_CPP})

set(SOURCE_RUN_SYSTEM faissSystem.cpp)
add_executable(faissSystem ${SOURCE_RUN_SYSTEM})

set(SOURCE_CAR_SYSTEM faissSystemCar.cpp)
add_executable(faissSystemCar ${SOURCE_CAR_SYSTEM})

set(SOURCE_CAR_IVF_SYSTEM faissSystemCarIVF.cpp)
add_executable(faissSystemCarIVF ${SOURCE_CAR_IVF_SYSTEM})

set(SOURCE_CAR_MAP_SYSTEM faissSystemMap.cpp)
add_executable(faissSystemMap ${SOURCE_CAR_MAP_SYSTEM})

set(SOURCE_SERVER server.cpp)
add_executable(Server ${SOURCE_SERVER})

set(SOURCE_MAP_SERVER mapServer.cpp)
add_executable(mapServer ${SOURCE_MAP_SERVER})

set(TEST_SORT_CP sort.cpp)
add_executable(sortcp ${TEST_SORT_CP})

target_link_libraries(
        extract_feature
        gpufaiss
        faiss
        libfeatureSql.so
        libfeature.so
        libssd.so
        libboost_system.so
        libboost_filesystem.so
        libopencv_core.so
        libopencv_imgcodecs.so
        libopencv_imgproc.so
        libopenblas.so.0
        libcudart.so
        libcublas.so
        libmysqlclient.so
        libmysqlpp.so
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
        libopenblas.so.0
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
        libopenblas.so.0
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
        libopenblas.so.0
        ${PROJECT_SOURCE_DIR}/../caffe-ssd/build/lib/libcaffe.so
        libglog.so
        libcudart.so
        libcublas.so
)

target_link_libraries(
        faissSystemMap
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
        libcudart.so
        libcublas.so
)

target_link_libraries(
        Server
        gpufaiss
        faiss
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
        mapServer
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
        test_pic
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
        sortcp
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