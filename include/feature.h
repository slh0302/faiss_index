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
    struct DPicture                                          //
    {
        int left, top, width, height;                               // 该feature 在原图中的坐标位置
        cv::Mat img;
        int label;
    };
    // define return file
    struct FileInfo
    {
        uchar* buff;
        long long size;
        FileInfo(uchar* buff ,long long size){
            this->buff = buff;
            this->size =size;
        }
    };
    /**
     *   @author Su
     *   @class FeatureIndex
     *   @usage 1. init network, extract or detect iamge feature
     *          2. return the feature
     */
    class FeatureIndex{
    private:
        // define const var
        const int batch_size = 1;
        // define private var
        Detector *det;
        std::string BLOB_NAME ;// "fc_hash/relu";
        caffe::Net<float> *feature_extraction_net;
        // define private func
        caffe::Net<float> *InitNet(std::string proto_file, std::string proto_weight);
    public:
        // define public mem and function
        // init function
        inline FeatureIndex(){ feature_extraction_net = NULL; det = NULL; }
        // only init feature
        FeatureIndex(std::string proto_file,std::string proto_mode);
        // once init all
        FeatureIndex(std::string feature_proto_file ,std::string feature_proto_weight,
                std::string det_proto_file, std::string det_proto_weight);
        // image file read
        FileInfo ImgRead(const char *file_name);
        // init gpu
        int InitGpu(const char *CPU_MODE, int GPU_ID);
        // feature extract from memory
        uchar* MemoryFeatureExtraction( std::vector<cv::Mat> pic_list, std::vector<int> label );
        // detect one Picture
        DPicture* DetectPicture(const char* picName, int label);
        // feature extract from picture
        template<typename Dtype>
        float* PictureFeatureExtraction(int count, std::string proto_file, std::string proto_weight);
        // float to binary
        uchar* floatToUnsignedChar(const float* data, int count);
    };
}
#endif //FAISS_INDEX_FEATURE_H
