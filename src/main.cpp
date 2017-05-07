#include <iostream>
#include <feature.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>

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
    for( int j =0 ;j< count ;j++)
    {
        for( int i = 0; i< 1024 ;i++){
            if(data[j*1024 + i]< 0.01){
                data[j*1024+i] = 0;
            }else if(data[j*1024 + i] > 1000){
                data[j*1024+i] = 1000;
            }
        }
    }
    //query
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 0);
    std::string proto_file = "/home/dell/CLionProjects/faiss_index/model/deploy_googlenet_hash.prototxt";
    std::string proto_weight = "/home/dell/CLionProjects/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(10,proto_file.c_str(), proto_weight.c_str(), "fc_hash/relu");
    std::cout<<"done extract"<<std::endl;
    for( int j =0 ;j< 10 ;j++)
    {
        for( int i = 0; i< 1024 ;i++){
            std::cout<<xq[j*1024+i]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }


    int d = 1024;                            // dimension
    // nq means num of query
    int nq = 5;
    int nlist = 1000;
    int k = 4;

    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search
    assert(!index.is_trained);
    index.train(count, data);
    assert(index.is_trained);
    index.add(count, data);

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        printf("I=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        index.nprobe = 10;
        index.search(nq, xq, k, D, I);

        printf("I=\n");
        for(int i = nq - 5; i < nq; i++) {
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