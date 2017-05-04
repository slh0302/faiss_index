//
// Created by dell on 17-5-4.
//
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include "feature.h"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"

caffe::Net<float>* feature_extraction_net = NULL;
Detector* det = NULL;
const int batch_size = 1;
const std::string BLOB_NAME = "fc_hash/relu";

int InitGpu(const char* CPU_MODE, int GPU_ID) {
    //GPU init
    if (CPU_MODE != NULL && (strcmp(CPU_MODE, "GPU") == 0)) {
        int device_id = 0;
        device_id = GPU_ID;
        Caffe::SetDevice(device_id);
        Caffe::set_mode(Caffe::GPU);
    }
    else {
        Caffe::set_mode(Caffe::CPU);
    }
    return 0;
}

Net<float>* InitNet(std::string PROTO_WEIGHT_PATH, std::string PROTO_FILE_PATH)
{
    //net work init
    std::string pretrained_binary_proto(PROTO_WEIGHT_PATH);
    std::string proto_file(PROTO_FILE_PATH);
    Net<float>* net(new Net<float>(proto_file, caffe::TEST));
    net->CopyTrainedLayersFrom(pretrained_binary_proto);
    return net;
}

unsigned char* img_read(const char* file_name) {
    FILE * fpPhoto;
    struct stat st1;
    stat(file_name, &st1);
    fpPhoto = fopen(file_name, "rb");
    unsigned char* buff = new unsigned char[st1.st_size];
    if (!fpPhoto)
    {
        printf("Unable to open file\n");
        return NULL;
    }
    long long total = 0;
    fread(buff, 1, st1.st_size, fpPhoto);
    fclose(fpPhoto);
    return buff;
}

char* feature_extraction(unsigned char * picYuvData, int width, int height, int format, int frameType, std::string frameIdx)
{
    //opencv read yuv
    std::vector<feature> f;
    IplImage* res = YUV420_To_IplImage_Opencv(picYuvData, width, height);
    cv::Mat img = cv::cvarrToMat(res);

    std::vector<cv::Mat> dv_list; // for feature extracting
    std::vector<int> dv_label;    // for feature extracting
    if( det == NULL || feature_extraction_net == NULL){
        std::cout<<"Detector or feature_extraction uninit! "<<std::endl;
        return f;
    }
    // begin detect
    char temp_s = '0';
    std::vector<std::vector<int> > ans = det->Detect(img);
    for (int i = 0; i < ans.size(); ++i) {
        // judge whether detected area is out of the side
        int width=0,height=0;
        if(ans[i][0]+ans[i][2]>img.cols){
            width = img.cols -ans[i][0];
        }else width = ans[i][2];

        if(ans[i][1]+ans[i][3]>img.rows){
            height = img.rows -ans[i][1];
        }else height = ans[i][3];
        cv::Rect rect(ans[i][0], ans[i][1], width, height);
        cv::Mat image_roi ;
        
        // init result
        feature temp;
        temp.left = ans[i][0];
        temp.top = ans[i][1];
        temp.width = ans[i][2];
        temp.height= ans[i][3];
        temp.frameIdx = frameIdx;
        temp.frameType = frameType;
        f.push_back(temp);

        cv::resize(img(rect), image_roi, cv::Size(224, 224));
        dv_list.push_back(image_roi);
        dv_label.push_back(atoi(frameIdx.c_str()));
    }

    // begin extract feature
    caffe::MemoryDataLayer<float> *m_layer = (caffe::MemoryDataLayer<float> *)feature_extraction_net->layers()[0].get();
    m_layer->AddMatVector(dv_list, dv_label);

    std::vector<caffe::Blob<float>*> input_vec;
    std::cout<<"ans size "<<ans.size()<<std::endl;
    int batch_count = ans.size() / batch_size;
    int dim_features;
    for (int index_size = 0; index_size<batch_count; ++index_size) {
        feature_extraction_net->Forward(input_vec);
        const boost::shared_ptr<Blob<float> > feature_blob = feature_extraction_net->blob_by_name(BLOB_NAME);
        dim_features = feature_blob->count() / batch_size;
        for (int n = 0; n < batch_size; ++n) {
            const float* feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
            unsigned char char_temp = 0;
            for (int d = 0; d < dim_features / ONEBYTESIZE; ++d) {
                unsigned char feature_temp = 0;
                for (int j = 0; j < ONEBYTESIZE; j++) {
                    if (feature_blob_data[d * ONEBYTESIZE + j] > 0.001) {
                        char_temp = 1;
                    }
                    else {
                        char_temp = 0;
                    }
                    feature_temp = feature_temp << 1;
                    feature_temp = feature_temp | char_temp;

                } // for (int j = 0; j < ONEBYTESIZE; j++)
                f[(n + index_size*batch_size)].binary_feature[d] = feature_temp;
            } // for (int d = 0; d < dim_features / ONEBYTESIZE; ++d)
        } // for (int n = 0; n < batch_size; ++n)
    } // for (int index_size = 0; index_size<batch_count; ++index_size)
    //judge batch szie
    bool isRemain = false;
    int remain = ans.size() - batch_count*batch_size;
    if (remain >0) {
        isRemain = true;
        feature_extraction_net->Forward(input_vec);
    }
    std::cout<<"isRemain: "<<isRemain<<std::endl;
    for (int n = 0; n < remain && isRemain; ++n) {//data new
        const boost::shared_ptr<Blob<float> > feature_blob = feature_extraction_net->blob_by_name(BLOB_NAME);
        const float* feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
        unsigned char char_temp = 0;
        for (int d = 0; d < dim_features / ONEBYTESIZE; ++d) {
            unsigned char feature_temp = 0;
            for (int j = 0; j < ONEBYTESIZE; j++) {
                if (feature_blob_data[d * ONEBYTESIZE + j]>0.001) {
                    char_temp = 1;
                }
                else {
                    char_temp = 0;
                }
                feature_temp = feature_temp << 1;
                feature_temp = feature_temp | char_temp;
            } // for (int j = 0; j < ONEBYTESIZE; j++)
            f[(batch_count*batch_size + n)].binary_feature[d] = feature_temp;
        }
    }
    return f;
}

IplImage* YUV420_To_IplImage_Opencv(unsigned char* pYUV420, int width, int height)
{
    if (!pYUV420)
    {
        return NULL;
    }

    IplImage *yuvimage, *rgbimg, *yimg, *uimg, *vimg, *uuimg, *vvimg;

    int nWidth = width;
    int nHeight = height;
    rgbimg = cvCreateImage(cvSize(nWidth, nHeight), IPL_DEPTH_8U, 3);
    yuvimage = cvCreateImage(cvSize(nWidth, nHeight), IPL_DEPTH_8U, 3);

    yimg = cvCreateImageHeader(cvSize(nWidth, nHeight), IPL_DEPTH_8U, 1);
    uimg = cvCreateImageHeader(cvSize(nWidth / 2, nHeight / 2), IPL_DEPTH_8U, 1);
    vimg = cvCreateImageHeader(cvSize(nWidth / 2, nHeight / 2), IPL_DEPTH_8U, 1);

    uuimg = cvCreateImage(cvSize(nWidth, nHeight), IPL_DEPTH_8U, 1);
    vvimg = cvCreateImage(cvSize(nWidth, nHeight), IPL_DEPTH_8U, 1);

    cvSetData(yimg, pYUV420, nWidth);
    cvSetData(uimg, pYUV420 + nWidth*nHeight, nWidth / 2);
    cvSetData(vimg, pYUV420 + long(nWidth*nHeight*1.25), nWidth / 2);
    cvResize(uimg, uuimg, CV_INTER_LINEAR);
    cvResize(vimg, vvimg, CV_INTER_LINEAR);

    cvMerge(yimg, uuimg, vvimg, NULL, yuvimage);
    cvCvtColor(yuvimage, rgbimg, CV_YCrCb2RGB);

    cvReleaseImage(&uuimg);
    cvReleaseImage(&vvimg);
    cvReleaseImageHeader(&yimg);
    cvReleaseImageHeader(&uimg);
    cvReleaseImageHeader(&vimg);

    cvReleaseImage(&yuvimage);

    if (!rgbimg)
    {
        return NULL;
    }

    return rgbimg;
}
