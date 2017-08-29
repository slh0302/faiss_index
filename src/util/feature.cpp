//
// Created by dell on 17-5-4.
//
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include "feature.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/db.hpp"

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
        Net<float>* net(new Net<float>(proto_model_file, caffe::TRAIN));
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

    /**
     *
     * @param picName
     * @return
     *
     */
    DPicture* FeatureIndex::DetectPicture(const char* picName, int label) {
        // always one input
        cv::Mat im = cv::imread(picName);
        std::vector<std::vector<int> > ans = det->Detect(im);
        DPicture* pic = new DPicture[ans.size()];
        for (int i = 0; i < ans.size(); ++i) {

            // judge whether detected area is out of the side
            int width=0,height=0;
            if(ans[i][0]+ans[i][2]>im.cols){
                width = im.cols -ans[i][0];
            }else width = ans[i][2];

            if(ans[i][1]+ans[i][3]>im.rows){
                height = im.rows -ans[i][1];
            }else height = ans[i][3];

            // store picture
            cv::Rect rect(ans[i][0], ans[i][1], width, height);
            pic[i].img = im(rect);
            pic[i].left = ans[i][0];
            pic[i].top = ans[i][1];
            pic[i].width = width;
            pic[i].height = height;

            // temp label info
            pic[i].label = label;
        }

        return pic;
    }

    /**
     *
     * @param count
     * @param proto_file
     * @param proto_weight
     * @return
     *
     */
    float* FeatureIndex::PictureFeatureExtraction(int count, std::string proto_file, std::string proto_weight, std::string blob_name) {

        std::string pretrained_binary_proto(proto_weight);
        std::string feature_extraction_proto(proto_file);

        BLOB_NAME = blob_name;
        Net<float>* _net(new Net<float>(feature_extraction_proto, caffe::TEST));
        _net->CopyTrainedLayersFrom(pretrained_binary_proto);

        std::string extract_feature_blob_names(BLOB_NAME);

        /////modify by su
        std::cout<<"batch size: "<< _net->blob_by_name(extract_feature_blob_names)->num()<<std::endl;
        int num_mini_batches = count / _net->blob_by_name(extract_feature_blob_names)->num();
        // init memory
        float* feature_dbs = new float[count * TOTALBYTESIZE ];
        std::vector<caffe::Blob<float>*> input_vec;
        Datum datum;
        const boost::shared_ptr<Blob<float> > feature_blob =
                _net->blob_by_name(extract_feature_blob_names);
        int batch_size = feature_blob->num();
        int dim_features = feature_blob->count() / batch_size;
        for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
            //std::cout<<"start"<<std::endl;
            _net->Forward(input_vec);
            const float* feature_blob_data;
            for (int n = 0; n < batch_size; ++n) {
                feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
                for (int d = 0; d < dim_features ; ++d) {
                    feature_dbs[d + (n + batch_index*batch_size)*dim_features] = feature_blob_data[d];
                } // for (int d = 0; d < dim_features / 8; ++d)
            }  // for (int n = 0; n < batch_size; ++n)
        }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

        // write the remain batch
        bool isRemain=false;
        int remain = count - num_mini_batches*(_net->blob_by_name(extract_feature_blob_names)->num());
        if(remain >0 ){
            isRemain=true;
            _net->Forward(input_vec);
        }
        if(isRemain){
            const float* feature_blob_data;
            for (int n = 0; n < remain; ++n) {//data new
                feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
                for (int d = 0; d < dim_features; ++d) {
                    feature_dbs[(num_mini_batches*batch_size+n)*dim_features +d] = feature_blob_data[d];
                } // for (int d = 0; d < dim_features/8; ++d)
            }  // for (int n = 0; n < remian; ++n)
        }  // for (int i = 0; i < num_features; ++i)

        std::cout<<"Successfully"<<std::endl;
        return feature_dbs;
    }

    /**
     * @param count
     *      file count
     * @param _net:
     *      caffe net
     * @return
     *      float result
     *
     */
    float* FeatureIndex::PictureFeatureExtraction(int count, caffe::Net<float> *_net, std::string blob_name) {
        std::string extract_feature_blob_names(blob_name);
        /////modify by su
        std::cout<<"batch size: "<< _net->blob_by_name(extract_feature_blob_names)->num()<<std::endl;
        int num_mini_batches = count / _net->blob_by_name(extract_feature_blob_names)->num();
        // init memory
        float* feature_dbs = new float[count * TOTALBYTESIZE ];
        std::vector<caffe::Blob<float>*> input_vec;
        Datum datum;
        const boost::shared_ptr<Blob<float> > feature_blob =
                _net->blob_by_name(extract_feature_blob_names);
        int batch_size = feature_blob->num();
        int dim_features = feature_blob->count() / batch_size;
        for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
            //std::cout<<"start"<<std::endl;
            _net->Forward(input_vec);
            const float* feature_blob_data;
            for (int n = 0; n < batch_size; ++n) {
                feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
                for (int d = 0; d < dim_features ; ++d) {
                    feature_dbs[d + (n + batch_index*batch_size)*dim_features] = feature_blob_data[d];
                } // for (int d = 0; d < dim_features / 8; ++d)
            }  // for (int n = 0; n < batch_size; ++n)
        }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

        // write the remain batch
        bool isRemain=false;
        int remain = count - num_mini_batches*(_net->blob_by_name(extract_feature_blob_names)->num());
        if(remain >0 ){
            isRemain=true;
            _net->Forward(input_vec);
        }
        if(isRemain){
            const float* feature_blob_data;
            for (int n = 0; n < remain; ++n) {//data new
                feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
                for (int d = 0; d < dim_features; ++d) {
                    feature_dbs[(num_mini_batches*batch_size+n)*dim_features +d] = feature_blob_data[d];
                } // for (int d = 0; d < dim_features/8; ++d)
            }  // for (int n = 0; n < remian; ++n)
        }  // for (int i = 0; i < num_features; ++i)

        std::cout<<"Successfully"<<std::endl;
        return feature_dbs;

    }


    float* FeatureIndex::MemoryPictureFeatureExtraction(int count, caffe::Net<float> *_net, std::string blob_name, std::vector<cv::Mat> pic_list, std::vector<int> label) {

        caffe::MemoryDataLayer<float> *m_layer = (caffe::MemoryDataLayer<float> *)_net->layers()[0].get();
        m_layer->AddMatVector(pic_list, label);

        std::string extract_feature_blob_names(blob_name);
        /////modify by su
        std::cout<<"batch size: "<< _net->blob_by_name(extract_feature_blob_names)->num()<<std::endl;
        int num_mini_batches = count / _net->blob_by_name(extract_feature_blob_names)->num();
        // init memory
        float* feature_dbs = new float[count * TOTALBYTESIZE ];
        std::vector<caffe::Blob<float>*> input_vec;
        Datum datum;
        const boost::shared_ptr<Blob<float> > feature_blob =
                _net->blob_by_name(extract_feature_blob_names);
        int batch_size = feature_blob->num();
        int dim_features = feature_blob->count() / batch_size;
        for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
            //std::cout<<"start"<<std::endl;
            _net->Forward(input_vec);
            const float* feature_blob_data;
            for (int n = 0; n < batch_size; ++n) {
                feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
                for (int d = 0; d < dim_features ; ++d) {
                    feature_dbs[d + (n + batch_index*batch_size)*dim_features] = feature_blob_data[d] ;
                } // for (int d = 0; d < dim_features / 8; ++d)
            }  // for (int n = 0; n < batch_size; ++n)
        }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

        // write the remain batch
        bool isRemain=false;
        int remain = count - num_mini_batches*(_net->blob_by_name(extract_feature_blob_names)->num());
        if(remain >0 ){
            isRemain=true;
            _net->Forward(input_vec);
        }
        if(isRemain){
            const float* feature_blob_data;
            for (int n = 0; n < remain; ++n) {//data new
                feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
                for (int d = 0; d < dim_features; ++d) {
                    feature_dbs[(num_mini_batches*batch_size+n)*dim_features +d] = feature_blob_data[d];
                } // for (int d = 0; d < dim_features/8; ++d)
            }  // for (int n = 0; n < remian; ++n)
        }  // for (int i = 0; i < num_features; ++i)

        std::cout<<"Successfully"<<std::endl;
        return feature_dbs;


    }


    /**
     * 
     * @param count
     * @param dq
     * @param _net
     * @param blob_name
     * @param pic_list
     * @param label
     */
    void FeatureIndex::MemoryPictureFeatureExtraction(int count, float* dq, caffe::Net<float> *_net, std::string blob_name, std::vector<cv::Mat> pic_list, std::vector<int> label) {

        caffe::MemoryDataLayer<float> *m_layer = (caffe::MemoryDataLayer<float> *)_net->layers()[0].get();
        m_layer->AddMatVector(pic_list, label);

        std::string extract_feature_blob_names(blob_name);
        /////modify by su
        std::cout<<"batch size: "<< _net->blob_by_name(extract_feature_blob_names)->num()<<std::endl;
        int num_mini_batches = count / _net->blob_by_name(extract_feature_blob_names)->num();
        // init memory
        float* feature_dbs = dq;
        std::vector<caffe::Blob<float>*> input_vec;
        Datum datum;
        const boost::shared_ptr<Blob<float> > feature_blob =
                _net->blob_by_name(extract_feature_blob_names);
        int batch_size = feature_blob->num();
        int dim_features = feature_blob->count() / batch_size;
        for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
            //std::cout<<"start"<<std::endl;
            _net->Forward(input_vec);
            const float* feature_blob_data;
            for (int n = 0; n < batch_size; ++n) {
                feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
                for (int d = 0; d < dim_features ; ++d) {
                    feature_dbs[d + (n + batch_index*batch_size)*dim_features] = feature_blob_data[d] ;
                } // for (int d = 0; d < dim_features / 8; ++d)
            }  // for (int n = 0; n < batch_size; ++n)
        }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

        // write the remain batch
        bool isRemain=false;
        int remain = count - num_mini_batches*(_net->blob_by_name(extract_feature_blob_names)->num());
        if(remain >0 ){
            isRemain=true;
            _net->Forward(input_vec);
        }
        if(isRemain){
            const float* feature_blob_data;
            for (int n = 0; n < remain; ++n) {//data new
                feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
                for (int d = 0; d < dim_features; ++d) {
                    feature_dbs[(num_mini_batches*batch_size+n)*dim_features +d] = feature_blob_data[d];
                } // for (int d = 0; d < dim_features/8; ++d)
            }  // for (int n = 0; n < remian; ++n)
        }  // for (int i = 0; i < num_features; ++i)

        std::cout<<"Successfully"<<std::endl;

    }


    /**
     *
     * @param data
     * @return
     *
     */
    uchar* FeatureIndex::floatToUnsignedChar(const float *data, int count) {
        int k = 0;
        int num_temp = TOTALBYTESIZE / ONEBYTESIZE;
        unsigned char* temp = new unsigned char[count * num_temp];
        while(k < count){
            for(int i=0; i< num_temp; i++){
                unsigned char temp_char = 0;
                for( int j = 0; j< ONEBYTESIZE ;j++){
                    temp_char = temp_char << 1;
                    if(data[k * TOTALBYTESIZE + i * ONEBYTESIZE + j] > 0.001){
                        temp_char = temp_char | 1;
                    }else{
                        temp_char = temp_char | 0;
                    }
                }
                temp[k * num_temp + i] = temp_char;
            }
            k++;
        }
        return temp;
    }

    /**
     *
     * @param filename
     *
     *
     */
    void FeatureIndex::InitLabelList(std::string filename) {
        std::ifstream in_label;
        in_label.open(filename, std::ios::in);
        for (int i = 0; i < 141756; i++){
            int name, data;
            in_label >> name >> data;
            LabelList.insert(std::pair<int, int>(name, data));
        }
        in_label.close();
    }

    /**
     *
     * @param end
     * @param label
     * @param info
     * @param index
     * @return
     *
     */
    double FeatureIndex::Evaluate(int end, int label, Info_String *info, long *index) {
        int num = 0;
        double res=0;
        std::vector< std::string > file_name_list;
        for (int i = 0; i < end; i++){
            boost::split(file_name_list, info[index[i]].info, boost::is_any_of(" ,!"), boost::token_compress_on);
            if (atoi(file_name_list[1].c_str()) == label){
                num++;
                res += num*1.0 / (i + 1);
            }
            file_name_list.clear();
        }
        res = res / LabelList[label];
        return res;
    }
}
