//
// Created by slh on 17-10-11.
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
    int count = 1;
    string index_filename = argv[2];

    Info_String* info = new Info_String[count];
    string temp;
    std::string ROOT_DIR = "/media/G/yanke/Vehicle_Data/wendeng_110/cropdata2/";
    std::string file_list_name ="/home/slh/caffe-ssd/detecter/examples/_temp/file_list";
    std::ofstream output(file_list_name,std::ios::out);
    for(int k =0; k<count; k++){
        strcpy(info[k].info, (ROOT_DIR + file_list).c_str());
        output<<info[k].info<<std::endl;
    }
    output.close();

    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);

    float* data ;
    int* color_re = new int[count];
    int* type_re = new int[count];

    /// int count, caffe::Net<float> * _net, std::string blob_name
    data = index.PictureAttrFeatureExtraction(count, net, "pool5/7x7_s1",
                                              "color/classifier", "model_loss1/classifier",
                                              color_re, type_re);
    /// PictureFeatureExtraction(count, net, "pool5/7x7_s1");

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
    float* data1 = new float[cou*1024];
    FILE* f = fopen (index_filename.c_str(),"rb");
    if(f == NULL){
        std::cout<<"File "<<index_filename<<" is not right"<<std::endl;
        return 0;
    }
    fread(data1,sizeof(float), cou*1024, f);
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
    int nq = 1;
    int k = 100;
    std::vector<faiss::Index::idx_t> nns (k * nq);
    std::vector<float>               dis (k * nq);
    indexPQ.setNumProbes(15);
    indexPQ.search(nq, data, k, dis.data(), nns.data());
    //index_person->search(nq, data, k, dis.data(), nns.data());

    /// SQL change
    FeatureSQL::FeatureSql sql;
    int row_count = 0 ;
    int* sql_result = sql.searchWithColor("", 0, row_count);

    int* total_res = new int[k];
    int hasNum = 0;
    for(int kk=0;kk<100;kk++){
        for(int i=0; i<row_count; i++){
            if( sql_result[i] == nns[kk] ){
                total_res[hasNum] = kk;
                hasNum++;
            }
        }

    }
    std::cout<<hasNum<<std::endl;
    delete data;
    return 0;
}
