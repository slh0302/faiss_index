//
// Created by slh on 17-5-9.
//

#include <iostream>
#include <feature.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    // init and check data
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
    //check data
    for( int j =0 ;j< 5 ;j++)
    {
        for( int i = 0; i< 1024 ;i++){
            std::cout<<data[j*1024+i]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    // query input
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 1);
    std::string proto_file = "/home/slh/faiss_index/model/deploy_googlenet_hash.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(10,proto_file.c_str(), proto_weight.c_str(), "fc_hash/relu");
    std::cout<<"done extract"<<std::endl;


    // There are two parameters to the search method:
    // nlist, the number of cells, and
    // nprobe, the number of cells (out of nlist)
    // that are visited to perform a search
    int d = 1024;                      // dimension
    int nq = 10;                       // nq means num of query
    int nlist = int(4 * sqrt(count));
    int k = 10;                        // k-NN

    faiss::IndexFlatL2 quantizer(d);   // the other index

    // The vectors are still stored in Voronoi cells,
    // but their size is reduced to a configurable number of bytes m
    // (d must be a multiple of m).
    // The vectors are still stored in Voronoi cells,
    // but their size is reduced to a configurable number of bytes m
    // (d must be a multiple of m).
    // when d = 64, m = 8 and float, Here we compress 64 32-bit floats to 8 bytes, 2048 bit ==> 64 bit
    // so the compression factor is 32.
    int m = 64;                        // in this place, d = 1024
                                       // compression factor is 16
    // definition
    // size_t d; equal to int d = 1024 ///< size of the input vectors
    // size_t M; equal to int m = 64   ///< number of subquantizers
    // size_t nbits; equal to 8        ///< number of bits per quantization index
    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);
    // # 8 specifies that each sub-vector is encoded as 8 bits
    double ttrain = elapsed();
    assert(!index.is_trained);
    index.verbose = true;
    index.train(count, data);
    printf("time: %.3f \n", elapsed()-ttrain);
    assert(index.is_trained);


//    { // I/O demo
//        const char *outfilename = "/home/slh/faiss_index/index_store/index_IVFPQ_NOTADD.faissindex";
//        printf ("[%.3f s] storing the pre-trained index to %s\n",
//                elapsed() - ttrain, outfilename);
//
//        write_index (&index, outfilename);
//    }


    index.add(count, data);

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        index.nprobe = 1024;

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
        printf("D=\n");
        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }


        delete [] I;
        delete [] D;
    }



    delete [] data;
    delete [] xq;

    return 0;
}