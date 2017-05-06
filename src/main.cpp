#include <iostream>
#include <feature.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>

int main(int argc, char** argv) {
    // input data
    if (argc <= 2){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 0;
    }
    int count = atoi(argv[2]);
    float* data = new float[count];
    FILE* f = fopen (argv[1],"rb");
    if(f == NULL){
        std::cout<<"File "<<argv[1]<<" is not right"<<std::endl;
        return 0;
    }
    fread(data,sizeof(float), count, f);
    fclose(f);
    std::cout<<"File read done"<<std::endl;

    //query
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 0);
    float * xq = input_index.PictureFeatureExtraction(1, "/home/dell/CLionProjects/faiss_index/model/deploy_googlenet_hash.prototxt",
                                          "/home/dell/CLionProjects/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel"  );

    int d = 1024;                            // dimension
//    int nb = 100000;                       // database size
//    int nq = 10000;                        // nb of queries

//    float *xb = new float[d * nb];
//    float *xq = new float[d * nq];
//
//    for(int i = 0; i < nb; i++) {
//        for(int j = 0; j < d; j++)
//            xb[d * i + j] = drand48();
//        xb[d * i] += i / 1000.;
//    }
//
//    for(int i = 0; i < nq; i++) {
//        for(int j = 0; j < d; j++)
//            xq[d * i + j] = drand48();
//        xq[d * i] += i / 1000.;
//    }
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