//
// Created by slh on 17-10-11.
//

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
    int GpuNum =  atoi(argv[4]);

    Info_String* info = new Info_String[count];
    string* s = new string[count];
    std::vector< std::string > file_name_list;
    string temp;
    std::string ROOT_DIR = "/media/G/yanke/Vehicle_Data/wendeng_110/cropdata2/";
    //std::string file_list_name ="/home/slh/caffe-ssd/detecter/examples/_temp/file_list";
    //std::ofstream output(file_list_name,std::ios::out);
    for(int k =0; k<count; k++){
        getline(infile, temp);
	    boost::split(file_name_list, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
        s[k] = ROOT_DIR + file_name_list[0];
        strcpy(info[k].info, (file_name_list[0]).c_str());
        //output<<s[k]<<std::endl;
    }
    infile.close();
    //output.close();

    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);


    float* data = new float[count];
    index.InitGpu("GPU", GpuNum);
   // data = index.PictureFeatureExtraction(count,net,"pool5/7x7_s1");

    int remain = count / 50 ; // remain
    std::vector<cv::Mat> pic_list;
    std::vector<int> label;
    index.InitGpu("GPU", GpuNum);
    for(int i=0; i< remain; i++){
        for (int j=0;j<50;j++){
            cv::Mat img = cv::imread(s[j+i*50]);
            cv::Mat cv_img ;
            cv::resize(img,cv_img, cv::Size(224,224));
            pic_list.push_back(cv_img);
            label.push_back(j);
        }
        index.MemoryPictureFeatureExtraction(count, &data[i*50], net, "pool5/7x7_s1", pic_list, label);
    }


    for(int k =0; k<10; k++){
        for(int j=0;j<1024;j++){
            std::cout<<data[j+k*1024]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }

    FILE * floatWrite = fopen(save_filename.c_str(),"wb");
    fwrite(data, sizeof(float), count * 1024, floatWrite);
    fclose(floatWrite);

    //Info_String* info_str = new Info_String[count];
    FILE* infoWrite =fopen((save_filename+"_info").c_str(),"wb");
    fwrite(info, sizeof(Info_String), count, infoWrite);
    fclose(infoWrite);

    delete data;
    return 0;
}
