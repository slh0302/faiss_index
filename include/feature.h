//
// Created by dell on 17-5-4.
//

#ifndef FAISS_INDEX_FEATURE_H
#define FAISS_INDEX_FEATURE_H
#include <iostream>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ssd.h"
namespace feature_index {

    #define TOTALBYTESIZE 1024
    #define ONEBYTESIZE 8
    // short cut the type of unsigned char
    typedef unsigned char uchar;
    // define return feature, can be modified
    struct feature                                                   // 一个feature特征
    {
        uchar binary_feature[TOTALBYTESIZE / ONEBYTESIZE]; // TOTALBYTESIZE = 1024 ，ONEBYTESIZE = 8
        int frameType;                                               // frameType 保存frameType值
        int left, top, width, height;                               // 该feature 在原图中的坐标位置
        std::string frameIdx;                                       // 表征原图的特征
    };

    // feature class
    class FeatureIndex{
        // define private var
    private:
        Detector *det;
        caffe::Net<float> *feature_extraction_net;
        caffe::Net<float> *InitNet(std::string PROTO_MODEL_PATH, std::string PROTO_FILE_PATH);
    public:
        // define public mem and function
        // init function
        inline FeatureIndex(){ feature_extraction_net = NULL; det = NULL; }
        //only init feature
        FeatureIndex(std::string proto_file,std::string proto_mode);
        //once init all
        FeatureIndex(std::string feature_proto_file ,std::string feature_proto_model,
                std::string det_proto_file, std::string det_proto_model);

        uchar *img_read(const char *file_name);
        int InitGpu(const char *CPU_MODE, int GPU_ID);
        // feature_extract
        uchar *feature_extraction(uchar *picYuvData, int width, int height, int format, int frameType,
                                 std::string frameIdx);

    };
}
#endif //FAISS_INDEX_FEATURE_H
