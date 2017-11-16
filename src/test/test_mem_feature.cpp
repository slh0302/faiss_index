//
// Created by slh on 17-11-15.
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

struct info_string
{
    std::string info;
};

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc,char** argv){
    google::InitGoogleLogging(argv[0]);
    FeatureIndex index = FeatureIndex();
    std::string proto_file = "/home/slh/faiss_index/model/deploy_google_multilabel.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";

    if(argc <= 1 ){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 1;
    }

    string file_list = argv[1];
    int count = atoi(argv[1]);
    string index_filename = argv[2];

    info_string* query_info = new info_string[count];
    int ori_count = 10000;
    info_string* ori_info = new info_string[ori_count];
    std::ifstream ins("crop_file", std::ios::in);
    int x,y,w,h;
    for(int i=0;i<ori_count;i++){
        ins>>ori_info[i].info;
    }
    ins.close();

    string temp;
    std::string ROOT_DIR = "/home/slh/Vechile_Search/libfaster_rcnn_cpp/bin/query/";
    std::string file_list_name ="/home/slh/caffe-ssd/detecter/examples/_temp/file_list";
    std::ofstream output(file_list_name,std::ios::out);
    std::ifstream input(file_list, std::ios::in);
    for(int k =0; k<count; k++){
        input>>query_info[k].info;
        output<<ROOT_DIR + query_info[k].info<<std::endl;
    }
    output.close();

    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);

    float* data ;

    /// int count, caffe::Net<float> * _net, std::string blob_name
    data = index.PictureFeatureExtraction(count, net, "pool5/7x7_s1");

    // TODO:: DATA
    int cou = ori_count;
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

    //faiss::gpu::
    {
        int ki = 100;
        int nq = count;
        long *I = new long[ki * nq];
        float *D = new float[ki * nq];
        // different from IndexIVF
        indexPQ.setNumProbes(20);
        double t0 = elapsed();
        indexPQ.search (nq, data, ki, D, I);
        double t1 = elapsed();
        for(int i =0;i<nq; i++) {
            std::string fn = query_info[i].info;
            std::stringstream ss ;
            ss<< i;
            ofstream re (fn + "_" + ss.str() + ".txt", std::ios::out);
            re<< query_info[i].info <<endl;
         
            // double resd1 = Evaluate2(ki, info_name[i].info, atoi(info[i].info), or_info, da+i*ki, index.getLabelList());
            for (int j = 0; j < ki; j++) {
                int _id_x = I[i*ki+j];
                std::string temp1 = ori_info[_id_x].info;
                re << temp1<<endl;
            }
            re.close();
        }

    }
    delete data;
    return 0;
}
