//
// Created by slh on 17-10-17.
//


#include <feature.h>
#include "binary.h"
#include "boost/algorithm/string.hpp"
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/index_io.h>
#include "featureSql.h"
#include <fstream>
#include <vector>

using namespace std;
using namespace feature_index;

#define DATA_BINARY 371
#define FAISS_GPU 10

/// Search one file :
///     1. extract feature
///     2. search (binary search)
///     3. Union

int main(int argc,char** argv){
    google::InitGoogleLogging(argv[0]);
    FeatureIndex index = FeatureIndex();
    std::string proto_file = "/home/slh/faiss_index/model/deploy_googlenet_multilabel_all.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";

    if(argc <= 1 ){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 1;
    }

    string file_list = argv[1];
    int count =  atoi(argv[2]);
    string index_filename = argv[3];

    Info_String* info = new Info_String[count];
    string temp;
    std::vector< std::string > file_name_list;

    /// Change picture root_dir to exec
    std::string ROOT_DIR = "/media/G/yanke/Vehicle_Data/wendeng_110/cropdata2/";
    std::string file_list_name ="/home/slh/caffe-ssd/detecter/examples/_temp/file_list";
    std::ofstream output(file_list_name,std::ios::out);
    std::ifstream input(file_list, std::ios::in);
    for(int k =0; k<count; k++){
        getline(input, temp);
        boost::split(file_name_list, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
        strcpy(info[k].info, (file_name_list[0] + " " + file_name_list[1]+ " "+ file_name_list[2]).c_str());
        output<<ROOT_DIR << info[k].info<<std::endl;
    }
    output.close();
    input.close();


    //Info_String* info_str = new Info_String[count];
    FILE* infoWrite =fopen((index_filename+"_info").c_str(),"wb");
    fwrite(info, sizeof(Info_String), count, infoWrite);
    fclose(infoWrite);
//
//    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);
//
//    int* color_re = new int[count];
//    int* type_re = new int[count];
////
////    /// int count, caffe::Net<float> * _net, std::string blob_name
////    index.PictureAttrExtraction(count, net,  "color/classifier",
////                                       "model_loss1/classifier", color_re, type_re);
//
//    std::ofstream o(index_filename, std::ios::out);
//    for(int kk=0; kk<count; kk++){
//        o<<kk<<","<<color_re[kk]<<","<<type_re[kk]<<std::endl;
//    }
//    o.close();

    return 0;
}
