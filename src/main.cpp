#include <iostream>
#include <feature.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

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
    fread(data,sizeof(float), count, f);
    fclose(f);
    std::cout<<"File read done"<<std::endl;
    //check data
    for( int j =0 ;j< 5 ;j++)
    {
        for( int i = 0; i< 1024 ;i++){
            std::cout<<data[j*1024+i]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    //query
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 0);
    std::string proto_file = "/home/dell/CLionProjects/faiss_index/model/deploy_googlenet_hash.prototxt";
    std::string proto_weight = "/home/dell/CLionProjects/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(10,proto_file.c_str(), proto_weight.c_str(), "fc_hash/relu");
    std::cout<<"done extract"<<std::endl;
    for( int j =0 ;j< 5 ;j++)
    {
        for( int i = 0; i< 1024 ;i++){
            std::cout<<xq[j*1024+i]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }


    int d = 1024;                            // dimension
    // nq means num of query
    int nq = 10;
    int nlist = 1000;
    int k = 10;

    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search

    // The vectors are still stored in Voronoi cells,
    // but their size is reduced to a configurable number of bytes m
    // (d must be a multiple of m).
    int m = 16;
    faiss::IndexIVFPQ index2(&quantizer, d, nlist, m, 8);


    assert(!index.is_trained);
    index.train(count, data);
    assert(index.is_trained);
    index.add(count, data);

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        double t0 = elapsed();
        index.search(nq, data, k, D, I);
        double t1 = elapsed();
        printf("time: %lf \n", t1-t0);
        printf("I=\n");
        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        index.nprobe = 10;

        double t2 = elapsed();
        index.search(nq, xq, k, D, I);
        double t3 = elapsed();
        printf("time: %lf \n", t3-t2);
        printf("I=\n");
        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }



    delete [] data;
    delete [] xq;

    return 0;
}