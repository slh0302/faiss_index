//
// Created by slh on 17-11-13.
//

#include <feature.h>
#include "binary.h"
#include <string>
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

int main(int argc,char** argv){
    google::InitGoogleLogging(argv[0]);
    FeatureIndex index = FeatureIndex();
    std::string proto_file = "/home/slh/faiss_index/model/deploy_googlenet_multilabel_all.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";

    if(argc <= 1 ){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 1;
    }
    ///  /home/slh/data/test/lb0054w/lb005-4w/list_data.txt
    string file_list = argv[1];
    int count = atoi(argv[2]);
    string index_filename = argv[3];
    int gpu_num = atoi(argv[4]);

    Info_String* info = new Info_String[count];
    string temp;
    std::vector< std::string > file_name_list;
    /// Change picture root_dir to exec
    std::string ROOT_DIR = "/home/slh/data/test/lb0054w/lb005-4w/lb005-4w/";
    std::string file_list_name ="/home/slh/caffe-ssd/detecter/examples/_temp/file_list";
    std::ofstream output(file_list_name,std::ios::out);
    std::ifstream input(file_list, std::ios::in);
    for(int k =0; k<count; k++){
        getline(input, temp);
        boost::split(file_name_list, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
        strcpy(info[k].info, (file_name_list[0]).c_str());
        output<<ROOT_DIR << file_name_list[0]<<std::endl;
    }
    output.close();
    input.close();

    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);

    float* data ;

    index.InitGpu("GPU",gpu_num);
    data = index.PictureFeatureExtraction(count, net, "pool5/7x7_s1");

    FILE* _f = fopen(index_filename.c_str(), "wb");
    fwrite(data, sizeof(float), count*1024, _f);
    fclose(_f);

    _f = fopen((index_filename+"_info").c_str(), "wb");
    fwrite(info, sizeof(Info_String), count, _f);
    fclose(_f);
    return 0;
}
