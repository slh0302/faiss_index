//
// Created by slh on 17-5-11.
//

#include <feature.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/index_io.h>

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

std::string proto_file = "/home/slh/faiss_index/model/deploy_person.prototxt";
std::string proto_weight = "/home/slh/faiss_index/model/model.caffemodel";

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
    faiss::gpu::GpuClonerOptions* options = new faiss::gpu::GpuClonerOptions();
    options->indicesOptions=faiss::gpu::INDICES_64_BIT;
    options->useFloat16CoarseQuantizer = false;
    options->useFloat16 = false;
    options->usePrecomputed = false;
    options->reserveVecs = 0;
    options->storeTransposed = false;
    options->verbose = true;
    faiss::gpu::StandardGpuResources resources;
    FILE *f = fopen (FileName, "r");
    if(f==NULL){
        std::cout<<"null done"<<std::endl;
    }
    faiss::Index* cpu_index = faiss::read_index(FileName, false);
    faiss::gpu::GpuIndexIVFPQ* index = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(
            faiss::gpu::index_cpu_to_gpu(&resources,0 ,cpu_index, options));

    std::cout<<"read done"<<std::endl;

    std::ofstream out("/home/slh/caffe-ssd/detecter/examples/_temp/file_list",std::ios::out);
    std::string ROOT_DIR = "/media/G/yanke/Vehicle_Data/wendeng_110/cropdata2/";
    for ( int i = 2;i<argc;i++){
        out<<ROOT_DIR + argv[i]<<" "<<i<<std::endl;
    }
    out.close();
    feature_index::FeatureIndex fea_index;
    float * xq = fea_index.PictureFeatureExtraction(argc - 2 ,proto_file.c_str(), proto_weight.c_str(), "loss3/feat_normalize");

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
            printf ("%7g ", dis[j + i * k]);
        }
        printf ("\n");
    }
    return 0;

}