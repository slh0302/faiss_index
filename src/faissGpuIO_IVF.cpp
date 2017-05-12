//
// Created by slh on 17-5-11.
//

//
// Created by slh on 17-5-11.
//

#include <feature.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/index_io.h>

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

std::string proto_file = "/home/slh/faiss_index/model/deploy_googlenet_hash.prototxt";
std::string proto_weight = "/home/slh/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel";

// load from index file, and search
int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    if( argc <= 2){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 0;
    }
    char* FileName = argv[1];
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
    faiss::gpu::GpuIndexIVFFlat* index = dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(
            faiss::gpu::index_cpu_to_gpu(&resources,2,cpu_index));

    std::cout<<"read done"<<std::endl;

    std::ofstream out("/home/slh/caffe-ssd/detecter/examples/_temp/file_list",std::ios::out);
    for ( int i = 2;i<argc;i++){
        out<<argv[i]<<" "<<i<<std::endl;
    }
    out.close();
    feature_index::FeatureIndex fea_index;
    float * xq = fea_index.PictureFeatureExtraction(argc - 2 ,proto_file.c_str(), proto_weight.c_str(), "fc_hash/relu");

    // para k-NN
    int k = 10;
    int nq = argc - 2;
    // result return
    std::vector<faiss::Index::idx_t> nns (k * nq);
    std::vector<float>               dis (k * nq);

    index->setNumProbes(1024);
    double t0 = elapsed();
    index->search(nq,xq,k,dis.data(),nns.data());
    double t1 = elapsed();
    printf("time: %.3f \n", t1-t0);


    // display data
    printf("I=\n");
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            printf ("%7ld ", nns[j + i * k]);
        }
        printf ("\n");
    }
    printf("D=\n");
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            printf ("%7.1g ", dis[j + i * k]);
        }
        printf ("\n");
    }
    return 0;

}