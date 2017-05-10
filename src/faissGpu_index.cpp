#include <feature.h>
#include <faiss/faiss.h>
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

int main(int argc, char** argv){
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
    fread(data,sizeof(float), count, f);
    fclose(f);
    std::cout<<"File read done"<<std::endl;

    // query input
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 1);
    std::string proto_file = "/home/slh/faiss_index/model/deploy_googlenet_hash.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(10,proto_file.c_str(), proto_weight.c_str(), "fc_hash/relu");
    std::cout<<"done extract"<<std::endl;

    int d = 1024;
    int ncentroids = int(4 * sqrt(count));
    faiss::gpu::StandardGpuResources resources;

    faiss::gpu::GpuIndexIVFPQ index (
            &resources, 1, d,
            ncentroids, 4, 8, true,
            faiss::gpu::INDICES_64_BIT,
            false,
            faiss::METRIC_L2);
    index.verbose = true;
    index.train(count, data);
    index.add (count, data);

    {
        int k = 10;
        int nq = 10;
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        // different from IndexIVF
        index.setNumProbes(2);
        double t0 = elapsed();
        index.search (nq, xq, k, D, I);
        double t1 = elapsed();
        printf("%5lf \n",t1 - t0 );
        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }


    }
	return 0;

}
