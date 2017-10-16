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
    int queryNum = 10;
    // input data
    if (argc <= 2){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 0;
    }
    int count = atoi(argv[2]);
    float* data = new float[count*1024];
    std::string filename = argv[1];

//    char s ='0';
//    for(int j=0; j<3;j++){
//        char temp = s+j;
//        std::string file_last = filename + "_" + temp;
//        std::cout<<file_last<<std::endl;
//        FILE* f = fopen (file_last.c_str(),"rb");
//        if(f == NULL){
//            std::cout<<"File "<<argv[1]<<" is not right"<<std::endl;
//            return 0;
//        }
//        fread(&data[j*2000000], sizeof(float), 2000000*1024, f);
//        fclose(f);
//        std::cout<<"file done "<<file_last<<std::endl;
//    }
    FILE* f = fopen (filename.c_str(),"rb");
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

    // query input
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 1);
    std::string proto_file = "/home/slh/faiss_index/model/deploy_person.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/model.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(queryNum, proto_file.c_str(), proto_weight.c_str(), "loss3/feat_normalize");
    std::cout<<"done extract"<<std::endl;

    // query info input
//    int* query = new int[queryNum];
//    std::ifstream inquery("/home/slh/faiss_index/model/queryinfo",std::ios::in);
//    for(int i=0; i<queryNum ;i++){
//        inquery>>query[i];
//    }
//    inquery.close();

    // Init label list
   // input_index.InitLabelList("/home/slh/faiss_index/model/labellist.txt");
    // There are two parameters to the search method:
    // nlist, the number of cells, and
    // nprobe, the number of cells (out of nlist)
    // that are visited to perform a search
    int d = 1024;                      // dimension
    int nq = queryNum;                       // nq means num of query
    int nlist = int(4 * sqrt(count));
    int k = 100;                        // max k-NN

    faiss::IndexFlatL2 quantizer(d);   // the other index

    // The vectors are still stored in Voronoi cells,
    // but their size is reduced to a configurable number of bytes m
    // (d must be a multiple of m).
    // The vectors are still stored in Voronoi cells,
    // but their size is reduced to a configurable number of bytes m
    // (d must be a multiple of m).
    // when d = 64, m = 8 and float, Here we compress 64 32-bit floats to 8 bytes, 2048 bit ==> 64 bit
    // so the compression factor is 32.
    int m = 32;                        // in this place, d = 1024
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


    { // I/O demo
        const char *outfilename = "/home/slh/faiss_index/index_store/index_CPU_personMap.faissindex";
        printf ("[%.3f s] storing the pre-trained index to %s\n",
                elapsed() - ttrain, outfilename);

        write_index (&index, outfilename);
    }


    //for(int i=0;i<9;i++){
    index.add(count, data);
    //}

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        index.nprobe = 15;
        //nq = 10;
        double t2 = elapsed();
        index.search(nq, xq, k, D, I);
        double t3 = elapsed();

//
//        // test for nprobe and k for mAP
//        // for nprobe, and k = 10
//        k = 10;
//        for (int tnprobe = 1; tnprobe < 128; tnprobe+=10){
//            index.nprobe = tnprobe;
//            double t2 = elapsed();
//            index.search(nq, xq, k, D, I);
//            double t3 = elapsed();
//
//            // Evaluate
//            double total_res = 0;
//            for(int i = 0; i< nq; i++){
//                double res = input_index.Evaluate(k, query[i], info, &I[i * k]);
//                total_res += res;
//            }
//            printf("mAP (nprobe: %d):  %7lf \n", tnprobe, total_res/nq);
//            printf("time: %lf \n", t3-t2);
//        }
//        // k-nn, and nprobe = 10
//        index.nprobe = 10;
//        for (int tk = 1; tk < 50; tk+=2){
//            double t2 = elapsed();
//            index.search(nq, xq, tk, D, I);
//            double t3 = elapsed();
//
//            // Evaluate
//            double total_res = 0;
//            for(int i = 0; i< nq; i++){
//                double res = input_index.Evaluate(tk, query[i], info, &I[i * tk]);
//                total_res += res;
//            }
//            printf("mAP (k-nn: %d):  %7lf \n", tk, total_res/nq);
//            printf("time: %lf \n", t3-t2);
//        }


        // evaluate
//        double total_res = 0;
//        for(int i = 0; i< nq; i++){
//            double res = input_index.Evaluate(k, query[i], info, &I[i * k]);
//            total_res += res;
//        }
       // printf("mAP:  %7lf\n", total_res/nq);

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