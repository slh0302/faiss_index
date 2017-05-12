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
    //query
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("CPU", 1);
    std::string proto_file = "/home/slh/faiss_index/model/deploy_googlenet_hash.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(10,proto_file.c_str(), proto_weight.c_str(), "fc_hash/relu");
    std::cout<<"done extract"<<std::endl;


    int d = 1024;                            // dimension
    // nq means num of query
    int nq = 10;
    int nlist = int(4 * sqrt(count));
    int k = 10;

    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search

    double ttrain = elapsed();
    assert(!index.is_trained);
    index.verbose =true;
    index.train(count, data);
    double ttdone = elapsed();
    printf("time: %lf \n", ttdone-ttrain);
    assert(index.is_trained);
    quantizer.verbose =true;
    quantizer.add(count, data);

    {       // search xq
        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);

        //index.nprobe = 10;
        double t2 = elapsed();
        quantizer.search(nq, xq, k, dis.data(), nns.data());
        double t3 = elapsed();
        printf("time: %lf \n", t3-t2);
        printf("I=\n");
        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", nns[i * k + j]);
            printf("\n");
        }

        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", dis[i * k + j]);
            printf("\n");
        }
    }



    delete [] data;
    delete [] xq;

    return 0;
}