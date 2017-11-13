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

#define DATA_BINARY 371
#define FAISS_GPU 10
double Evaluate(int end, int label, Info_String *info, long *index, std::map<int, int>LabelList) {
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
double Evaluate2(int end, std::string name, int label, Info_String *info, long *index, std::map<int, int>LabelList) {
    int num = 0;
    double res=0;
    std::vector< std::string > file_name_list;
    for (int i = 0; i < end; i++){
        boost::split(file_name_list, info[index[i]].info, boost::is_any_of(" ,!"), boost::token_compress_on);
        if(name != file_name_list[0]){
            if (atoi(file_name_list[1].c_str()) == label){
                num++;
                res += num*1.0 / (i + 1);
            }
        }
        file_name_list.clear();
    }
    res = res / (LabelList[label]-1);
    return res;
}
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
    Info_String* info_name = new Info_String[count];
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
        strcpy(info[k].info, (file_name_list[1]).c_str());
        strcpy(info_name[k].info, (file_name_list[0]).c_str());
        output<<ROOT_DIR << file_name_list[0]<<std::endl;
    }
    output.close();
    input.close();


    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);

    float* data ;

    index.InitGpu("GPU",gpu_num);
    data = index.PictureFeatureExtraction(count, net, "pool5/7x7_s1");

    int cou = 1080000;
    int* list_total=new int[400000];
    std::ifstream in1("/home/slh/labellist.txt", std::ios::in);
    int num1,num2;
    while(in1>>num1>>num2){
        list_total[num1] = num2;
    }
    in1.close();

    float* data1 = new float[cou*1024];
    FILE* f = fopen (index_filename.c_str(),"rb");
    if(f == NULL){
        std::cout<<"File "<<index_filename<<" is not right"<<std::endl;
        return 0;
    }
    fread(data1,sizeof(float), cou*1024, f);
    fclose(f);

    Info_String* or_info = new Info_String[cou];
    f = fopen((index_filename+"_info").c_str(),"rb");
    if(f == NULL){
        std::cout<<"File "<<index_filename<<"_info is not right"<<std::endl;
        return 0;
    }
    fread(or_info,sizeof(Info_String), cou, f);
    fclose(f);

    int ncentroids = int(4 * sqrt(cou));
    faiss::gpu::StandardGpuResources resources;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = 0;
    std::cout<< "ncentroids: "<<ncentroids <<std::endl;
    faiss::gpu::GpuIndexIVFPQ indexPQ (
            &resources, 1024,
            ncentroids, 32, 8,
            faiss::METRIC_L2,config);
    indexPQ.verbose = true;
    indexPQ.train(cou, data1);
    std::cout<<"done"<<std::endl;

    indexPQ.add (cou, data1);
    int ki = 200;

    /// result return
    int nq = count;
    index.InitLabelList("/home/slh/faiss_index/model/labellist.txt");
    std::vector<faiss::Index::idx_t> nns (ki * nq);
    std::vector<float>               dis (ki * nq);
    std::vector<std::string> info_list;
    indexPQ.setNumProbes(30);
    indexPQ.search(nq, data, ki, dis.data(), nns.data());
    long* da = nns.data();
    for(int i =0;i<nq;i++) {
        std::vector<std::string> OUT_LIST;
        boost::split(OUT_LIST, info_name[i].info, boost::is_any_of("."), boost::token_compress_on);
        ofstream re (OUT_LIST[0] + ".txt",std::ios::out);
        re<< info_name[i].info <<endl;
        double resd = Evaluate(ki, atoi(info[i].info), or_info, da+i*ki, index.getLabelList());
        // double resd1 = Evaluate2(ki, info_name[i].info, atoi(info[i].info), or_info, da+i*ki, index.getLabelList());
        re<< resd << endl;
        for (int j = 0; j < ki; j++) {
            int _id_x = nns[i*ki+j];
            std::string temp1 = or_info[_id_x].info;
            boost::split(info_list, temp1, boost::is_any_of(" ,!"), boost::token_compress_on);
            re << info_list[0] << endl;
        }
        re.close();
    }

    delete data;
    return 0;
}
