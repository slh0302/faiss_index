//
// Created by slh on 17-10-27.
//

#include <feature.h>
#include "boost/algorithm/string.hpp"
#include <fstream>
using namespace std;
using namespace feature_index;


int main(int argc,char** argv){
    google::InitGoogleLogging(argv[0]);
    FeatureIndex index = FeatureIndex();
    /// change path
    std::string proto_file = "/home/slh/faiss_index/model/deploy_googlenet_multilabel_all.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";

    if(argc <= 1 ){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 1;
    }

    string file_list = argv[1];
    int count =  atoi(argv[2]);
    string index_filename = argv[3];
    std::vector< std::string > file_name_list;
    Info_String* info = new Info_String[count];
    string temp;

    /// Change picture root_dir to exec
    std::string ROOT_DIR = "/home/slh/data/test/new/";
//    std::string file_list_name ="/home/slh/retrieval/model/file_list";
//    std::ofstream output(file_list_name,std::ios::out);
    std::ifstream input(file_list, std::ios::in);
    std::vector<cv::Mat> pic_list;
    std::vector<int> label;
    for(int k =0; k<count; k++){
        getline(input, temp);
        boost::split(file_name_list, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
        strcpy(info[k].info, (ROOT_DIR + file_name_list[0]).c_str());
        cv::Mat cv_origin = cv::imread(info[k].info, CV_8U);
        cv::Mat cv_img ;
        cv::resize(cv_origin,cv_img, cv::Size(224,224));
        pic_list.push_back(cv_img);
        label.push_back(0);

    }
//    output.close();
    input.close();
    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);

    int* color_re = new int[count];
    int* type_re = new int[count];

    /// int count, caffe::Net<float> * _net, std::string blob_name
    float* data = index.PictureAttrFeatureExtraction(count, net, "pool5/7x7_s1",
                                                     "color/classifier", "model_loss1/classifier",
                                                     color_re, type_re, pic_list, label);

    std::ofstream o(index_filename, std::ios::out);
    for(int kk=0; kk<count; kk++){
        o<<kk<<","<<color_re[kk]<<","<<type_re[kk]<<std::endl;
        for(int i=0;i<1024;i++){
            o<< i<<" "<<data[kk*1024 + i]<<endl;
        }

    }
    o.close();

    return 0;
}
