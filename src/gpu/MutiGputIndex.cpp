//
// Created by slh on 17-9-2.
//


#include <feature.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/IndexProxy.h>
#include <faiss/gpu/GpuIndexFlat.h>
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
    //   std::string filename = argv[1];
    // file read data
    FILE* f = fopen (argv[1],"rb");
    if(f == NULL){
        std::cout<<"File "<<argv[1]<<" is not right"<<std::endl;
        return 0;
    }
    fread(data,sizeof(float), count*1024, f);
    fclose(f);

    // query input
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 14);
    std::string proto_file = "/home/slh/faiss_index/model/deploy_google_multilabel.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(10,proto_file.c_str(), proto_weight.c_str(), "pool5/7x7_s1");
    std::cout<<"done extract"<<std::endl;

    // test
    for(int i =0 ;i< 10;i++){
        for(int j=0 ;j<1024;j++){
            std::cout<<xq[i*1024+j]<<" ";
        }
        std::cout<<std::endl;
    }
    int d = 1024;

    // 4*sqrt(count)
    int ncentroids = int(4 * sqrt(count));

    int ngpus = 5;
    faiss::gpu::StandardGpuResources* res[ngpus];
    for(int j=0;j<ngpus;j++){
        faiss::gpu::StandardGpuResources* resources =
                new faiss::gpu::StandardGpuResources();
        res[j] = resources;
    }
    // Gpu Indexes
    int gpuArray[ngpus] = {4,5,6,8,13};
    faiss::gpu::GpuIndexIVFPQ* indexes[ngpus];
    for(int k = 0; k < ngpus; k++){
        faiss::gpu::GpuIndexIVFPQ* index = new faiss::gpu::GpuIndexIVFPQ(res[k], gpuArray[k], d,
                                                                         ncentroids, 32, 8, true,
                                                                         faiss::gpu::INDICES_64_BIT,
                                                                         false,
                                                                         faiss::METRIC_L2);
        index->setNumProbes(128);
        indexes[k] = index;
        index->verbose = true;
    }
    faiss::gpu::IndexProxy indexPro;
    for(int z = 0; z < ngpus; z++){
        indexPro.addIndex(indexes[z]);
        //index->verbose = true;
    }
    indexPro.verbose = true;
    indexPro.train(count, data);
    indexPro.add(count, data);
    std::cout<<"done"<<std::endl;


//    { // I/O demo
//        const char *outfilename = "/home/slh/faiss_index/index_store/index_person_map_nodata.faissindex";
//        faiss::Index * cpu_index = faiss::gpu::index_gpu_to_cpu (&index);
//        write_index (cpu_index, outfilename);
//        printf ("done save \n");
//        delete cpu_index;
//
    //faiss::gpu::
    {
        int k = 10;
        int nq = 10;
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        // different from IndexIVF
        double t0 = elapsed();
        indexPro.search (nq, xq, k, D, I);
        double t1 = elapsed();
        printf("%7lf \n",t1 - t0 );
        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%7ld ", I[i * k + j]);
            printf("\n");
        }
        printf("D=\n");
        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

    }
    return 0;

}
