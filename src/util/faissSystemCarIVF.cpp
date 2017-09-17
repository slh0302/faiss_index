//
// Created by slh on 17-5-14.
//

// data need to be saved in
// /home/slh/pro/run/runResult/result.txt

//
// Created by slh on 17-5-11.
//

#include <feature.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/index_io.h>
#include "boost/algorithm/string.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define DATA_COUNT 43455
double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

struct Info_String
{
    char info[100];
};

std::string proto_file = "/home/slh/faiss_index/model/deploy_google_multilabel.prototxt";
std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";

// load from index file, and search
int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    if( argc <= 4){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 0;
    }
    char* FileName = argv[1];
    int GpuNum = atoi(argv[2]);
    int Limit = atoi(argv[3]);
    // file read
    Info_String* info = new Info_String[DATA_COUNT];
    FILE* fin = fopen("/home/slh/data/data_car_color_43455_info","rb");
    fread(info, sizeof(Info_String),DATA_COUNT,fin);
    fclose(fin);
    // read index
    // index_cpu_to_gpu() fourth para == defaults
    // GpuClonerOptions::GpuClonerOptions():
    //    indicesOptions(INDICES_64_BIT),
    //            useFloat16CoarseQuantizer(false),
    //            useFloat16(false),
    //            usePrecomputed(true),
    //            reserveVecs(0),
    //            storeTransposed(false),
    //            verbose(0)

    faiss::gpu::StandardGpuResources resources;
    faiss::Index* cpu_index = faiss::read_index(FileName, false);
    faiss::gpu::GpuIndexIVF* index = dynamic_cast<faiss::gpu::GpuIndexIVF*>(
            faiss::gpu::index_cpu_to_gpu(&resources,GpuNum,cpu_index));

    std::cout<<"read done"<<std::endl;

    std::ofstream out("/home/slh/caffe-ssd/detecter/examples/_temp/file_list",std::ios::out);
    for ( int i = 4;i<argc;i++){
        out<<argv[i]<<" "<<i<<std::endl;
    }
    out.close();
    feature_index::FeatureIndex fea_index;
    fea_index.InitGpu("GPU", 1);
    float * xq = fea_index.PictureFeatureExtraction(argc - 4 ,proto_file.c_str(), proto_weight.c_str(), "pool5/7x7_s1");

    // para k-NN
    int k = 20;
    int nq = argc - 4;
    // result return
    std::vector<faiss::Index::idx_t> nns (k * nq);
    std::vector<float>               dis (k * nq);

    index->setNumProbes(Limit);
    double t0 = elapsed();
    index->search(nq,xq,k,dis.data(),nns.data());
    double t1 = elapsed();
    //printf("time: %.3f \n", t1-t0);

    // output the result
    std::ofstream reout("/home/slh/pro/run/runResult/result.txt",std::ios::out);
    std::vector<std::string> file_name_list;
    std::string root_dir = "/media/vehicle_org/wengdeng/1/";
    reout<<t1 - t0<<std::endl;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            std::string temp = info[nns[j + i * k]].info;
            //std::cout<<temp<<std::endl;
            if(temp.length() < 5){
                continue;
            }
            boost::split(file_name_list, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
            cv::Mat im = cv::imread(root_dir + file_name_list[0]);
            int x = atoi(file_name_list[1].c_str());
            int y = atoi(file_name_list[2].c_str());
            int width = atoi(file_name_list[3].c_str());
            int height = atoi(file_name_list[4].c_str());
            //std::cout<<x<<y<<width<<height<<std::endl;
            rectangle(im,cvPoint(x,y),cvPoint(x+width, y+height),cv::Scalar(0,0,255),3,1,0);
            //out im
            IplImage qImg;
            qImg = IplImage(im); // cv::Mat -> IplImage
            char stemp[200];
            std::string str_name = argv[4];
            int index_slash = str_name.find_last_of('/');
            int index_dot = str_name.find_last_of('.');
            str_name = str_name.substr(index_slash+1,index_dot- index_slash-1);
            sprintf(stemp,"/home/slh/pro/run/originResult/%s_%d.jpg",str_name.c_str(),j);
            cvSaveImage(stemp,&qImg);
            reout << stemp <<std::endl;
        }
    }
    reout.close();
    printf("Done");
    return 0;

}