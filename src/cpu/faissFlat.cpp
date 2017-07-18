//
// Created by slh on 17-6-12.
//

#include <iostream>
#include <fstream>
#include <feature.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    int queryNum = 2000;
    // input data
    if (argc <= 2){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 0;
    }
    int count = atoi(argv[2]);
    float* data = new float[count*1024];
    FILE* f = fopen (argv[1],"rb");
    if(f == NULL){
        std::cout<<"File "<<argv[1]<<" is not right"<<std::endl;
        return 0;
    }
    fread(data,sizeof(float), count*1024, f);
    fclose(f);
    std::cout<<"File read done"<<std::endl;

    // file read info
    std::string file_info = std::string(argv[1])+"_info";
    FILE* f2 = fopen(file_info.c_str(),"rb");
    feature_index::Info_String* info = new feature_index::Info_String[count];
    fread(info, sizeof(feature_index::Info_String), count, f2);
    fclose(f2);
    std::cout<<"File info read done"<<std::endl;

    //query
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("CPU", 1);
    std::string proto_file = "/home/slh/faiss_index/model/deploy_google_multilabel.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(queryNum, proto_file.c_str(), proto_weight.c_str(), "pool5/7x7_s1");
    std::cout<<"done extract"<<std::endl;

    // query info input
    int* query = new int[queryNum];
    std::ifstream inquery("/home/slh/faiss_index/model/queryinfo",std::ios::in);
    for(int i=0; i<queryNum ;i++){
        inquery>>query[i];
    }
    inquery.close();

    // Init label list
    input_index.InitLabelList("/home/slh/faiss_index/model/labellist.txt");


    int d = 1024;                            // dimension
    // nq means num of query
    int nq = queryNum;
    int nlist = int(4 * sqrt(count));
    int k = 100;                          // max k-nn

    faiss::IndexFlatL2 index(d);       // the other index
    // here we specify METRIC_L2, by default it performs inner-product search

    // for(int i=0;i<9;i++){
    index.add(count, data);
    // }


    {       // search xq
        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);

//        index.nprobe = 10;
//        double t2 = elapsed();
//        index.search(nq, xq, k, dis.data(), nns.data());
//        double t3 = elapsed();

        // test for nprobe and k for mAP
        // for nprobe, and k = 10
        k = 10;


        // test for k
        // index.nprobe = 10;
        for (int tk = 1; tk < 50; tk+=2){
            double t2 = elapsed();
            index.search(nq, xq, tk, dis.data(), nns.data());
            double t3 = elapsed();
            // Evaluate
            double total_res = 0;
            for(int i = 0; i< nq; i++){
                double res = input_index.Evaluate(tk, query[i], info, &nns[i * tk]);
                total_res += res;
            }
            printf("mAP (k-nn: %d):  %7lf \n", tk, total_res/nq);
            printf("time: %lf \n", t3-t2);
        }

        // for k
//        // Evaluate
//        double total_res = 0;
//        for(int i = 0; i< nq; i++){
//            double res = input_index.Evaluate(k, query[i], info, &nns[i * k]);
//            total_res += res;
//        }
//        printf("mAP:  %7lf\n", total_res/nq);

//        printf("time: %lf \n", t3-t2);
//        printf("I=\n");
//        for(int i = 0; i < nq; i++) {
//            for(int j = 0; j < k; j++)
//                printf("%5ld ", nns[i * k + j]);
//            printf("\n");
//        }
//
//        for(int i = 0; i < nq; i++) {
//            for(int j = 0; j < k; j++)
//                printf("%7g ", dis[i * k + j]);
//            printf("\n");
//        }
    }



    delete [] data;
    delete [] xq;

    return 0;
}