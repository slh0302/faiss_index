//
// Created by slh on 17-5-10.
//

#include <feature.h>
#include "boost/algorithm/string.hpp"
#include <fstream>
#include <vector>
// extract feature
using namespace std;
using namespace feature_index;

struct Info_String
{
    char info[100];
};

int main(int argc,char** argv){
    google::InitGoogleLogging(argv[0]);
    FeatureIndex index = FeatureIndex();
    std::string proto_file = "/home/slh/faiss_index/model/deploy_google_multilabel_memory.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";

    if(argc <= 1 ){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 1;
    }

    string file_list = argv[1];
    ifstream infile(file_list.c_str(), ios::in);
    int count = atoi(argv[2]);
    string save_filename = argv[3];

    Info_String* info = new Info_String[count];
    string* s = new string[count];
    std::vector< std::string > file_name_list;
    string temp;
    std::string ROOT_DIR = "/media/vehicle_res/person/out/";
    for(int k =0; k<count; k++){
        getline(infile, temp);
        boost::split(file_name_list, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
        s[k] = ROOT_DIR + file_name_list[0];
        strcpy(info[k].info, (file_name_list[1]+" "+ file_name_list[2] +" "+
                                  file_name_list[3] +" "+file_name_list[4]+" "+file_name_list[5] + " " + file_name_list[6]).c_str());

    }
    infile.close();


    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);

    float* data = new float[count]; // data
    int remain = count / 200 ; // remain
    std::vector<cv::Mat> pic_list;
    std::vector<int> label;
    index.InitGpu("GPU", 11);
    for(int i=0; i< remain; i++){
        for (int j=0;j<200;j++){
            cv::Mat img = cv::imread(s[j+i*200]);
            cv::Mat cv_img ;
            cv::resize(img,cv_img, cv::Size(224,224));
            pic_list.push_back(cv_img);
            label.push_back(j);
        }
        index.MemoryPictureFeatureExtraction(count, &data[i*200], net, "pool5/7x7_s1", pic_list, label);
    }


    FILE * floatWrite = fopen(save_filename.c_str(),"wb");
    fwrite(data, sizeof(float), count * 1024, floatWrite);
    fclose(floatWrite);

    //Info_String* info_str = new Info_String[count];
    FILE* infoWrite =fopen((save_filename+"_info").c_str(),"wb");
    fwrite(info, sizeof(Info_String), count, infoWrite);
    fclose(infoWrite);

    delete data;
    delete s;
    return 0;
}