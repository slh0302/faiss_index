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

namespace feature_index{

    /**
     *
     * @param proto_file
     * @param proto_weight
     * @return caffe::Net<float>*
     * @usage  1. init the network
     *
     */
    caffe::Net<float>* FeatureIndex::InitNet(std::string proto_file, std::string proto_weight) {
        //net work init
        std::string pretrained_binary_proto(proto_weight);
        std::string proto_model_file(proto_file);
        Net<float>* net(new Net<float>(proto_model_file, caffe::TEST));
        net->CopyTrainedLayersFrom(pretrained_binary_proto);
        return net;
    }

    /**
     *
     * @param proto_file
     * @param proto_weight
     *
     */
    FeatureIndex::FeatureIndex(std::string proto_file, std::string proto_weight) {
        feature_extraction_net = InitNet(proto_file, proto_weight);
    }

    /**
     *
     * @param feature_proto_file
     * @param feature_proto_weight
     * @param det_proto_file
     * @param det_proto_weight
     *
     */
    FeatureIndex::FeatureIndex(std::string feature_proto_file, std::string feature_proto_weight,
                               std::string det_proto_file, std::string det_proto_weight) {
        feature_extraction_net = InitNet(feature_proto_file, feature_proto_weight);
        det = new Detector(det_proto_file, det_proto_weight);
    }

    /**
     *
     * @param file_name
     * @return
     *
     */
    FileInfo FeatureIndex::ImgRead(const char *file_name) {
        FILE * fpPhoto;
        struct stat st1;
        stat(file_name, &st1);
        fpPhoto = fopen(file_name, "rb");
        unsigned char* buff = new unsigned char[st1.st_size];
        if (!fpPhoto)
        {
            printf("Unable to open file\n");
            return FileInfo(NULL, 0);
        }
        long long total = st1.st_size;
        fread(buff, 1, st1.st_size, fpPhoto);
        fclose(fpPhoto);
        return FileInfo(buff, total);
    }

    /**
     *
     * @param CPU_MODE
     * @param GPU_ID
     * @return
     *
     */
    int FeatureIndex::InitGpu(const char *CPU_MODE, int GPU_ID) {
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

    /**
     *
     * @param picData
     * @param width
     * @param height
     * @return
     *
     */
    uchar* FeatureIndex::MemoryFeatureExtraction(std::vector<cv::Mat> pic_list, std::vector<int> label ) {
        // begin extract feature
        caffe::MemoryDataLayer<float> *m_layer = (caffe::MemoryDataLayer<float> *)feature_extraction_net->layers()[0].get();
        m_layer->AddMatVector(pic_list, label);
        std::vector<caffe::Blob<float>*> input_vec;
        // open size
        int pic_size = pic_list.size();
        uchar* f = new uchar[pic_size * TOTALBYTESIZE / ONEBYTESIZE];

        std::cout<<"ans size "<<pic_list.size()<<std::endl;
        int batch_count = pic_size / batch_size;
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
                    f[(n + index_size*batch_size) * TOTALBYTESIZE / ONEBYTESIZE + d] = feature_temp;
                } // for (int d = 0; d < dim_features / ONEBYTESIZE; ++d)
            } // for (int n = 0; n < batch_size; ++n)
        } // for (int index_size = 0; index_size<batch_count; ++index_size)
        //judge batch szie
        bool isRemain = false;
        int remain = pic_size - batch_count*batch_size;
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
                f[(batch_count*batch_size + n) * TOTALBYTESIZE / ONEBYTESIZE + d] = feature_temp;
            }
        }
        return f;

    }

    int FeatureIndex::DetectPicture(int argc, char **argv) {

    }
}
