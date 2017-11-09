//
// Created by slh on 17-10-11.
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
    int count = atoi(argv[2]);
    string index_filename = argv[3];
    int gpu_num = atoi(argv[4]);
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
        strcpy(info[k].info, (file_name_list[1]).c_str());
        output<<ROOT_DIR << file_name_list[0]<<std::endl;
    }
    output.close();
    input.close();


    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);

    float* data ;

//    /// int count, caffe::Net<float> * _net, std::string blob_name
//    data = index.PictureAttrFeatureExtraction(count, net, "pool5/7x7_s1",
//                                              "color/classifier", "model_loss1/classifier",
//                                              color_re, type_re);
    index.InitGpu("GPU",gpu_num);
    data = index.PictureFeatureExtraction(count, net, "pool5/7x7_s1");

    /**
    //    /// binary change
    //    /// Init Binary Index
    //    void * p = FeatureBinary::CreateIndex(0);
    //
    //    /// Load Binary Table
    //    std::string table_filename="/home/slh/faiss_index/index_store/table.index";
    //    if(!std::fstream(table_filename.c_str())) {
    //        std::cout << "Table File Wrong" << std::endl;
    //        return 1;
    //    }
    //    FeatureBinary::CreateTable(table_filename.c_str(), 16);
    //
    //    /// Load Binary Index
    //
    //    /// TODO:: Index File Name Change
    //    std::string IndexFileName("/home/slh/data/demo/data_binary_map_371");
    //    std::string IndexInfoName = IndexFileName + "_info";
    //    FeatureBinary::LoadIndex(p, IndexFileName.c_str(), IndexInfoName.c_str(), DATA_BINARY);
    //
    **/

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
    config.device = 9;
    std::cout<< "ncentroids: "<<ncentroids <<std::endl;
    faiss::gpu::GpuIndexIVFPQ indexPQ (
            &resources, 1024,
            ncentroids, 32, 8,
            faiss::METRIC_L2,config);
    indexPQ.verbose = true;
    indexPQ.train(cou, data1);
    std::cout<<"done"<<std::endl;

    indexPQ.add (cou, data1);

    /// result return
    int nq = 2000;
    for( int ki=10;ki<400; ki+=10){
        int k = ki;
        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);
        std::vector< std::string > info_list;
        indexPQ.setNumProbes(15);
        indexPQ.search(nq, data, k, dis.data(), nns.data());

        double totalRecall = 0;
        for(int j=0;j<nq;j++){
            int q_temp = atoi(info[j].info);
            int times = 0;
            for(int t=0;t<ki;t++){
                int _id_x = nns[j*ki+t];
                std::string temp1 =or_info[_id_x].info;
                boost::split(info_list,temp1 , boost::is_any_of(" ,!"), boost::token_compress_on);
                if(q_temp == atoi(info_list[1].c_str())){
                    times++;
                }
                totalRecall += times / list_total[q_temp];
            }
        }
        cout<<"Top "<<ki <<" "<<totalRecall/nq<<endl;
        //index_person->search(nq, data, k, dis.data(), nns.data());
    }


//    /// SQL change
//    FeatureSQL::FeatureSql sql;
//    int row_count = 0 ;
//    int* sql_result = sql.searchWithColor("", 0, row_count);
//
//    int* total_res = new int[k];
//    int hasNum = 0;
//    for(int kk=0;kk<100;kk++){
//        for(int i=0; i<row_count; i++){
//            if( sql_result[i] == nns[kk] ){
//                total_res[hasNum] = kk;
//                hasNum++;
//            }
//        }
//
//    }
//    std::cout<<hasNum<<std::endl;
    delete data;
    return 0;
}
