#include <feature.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/index_io.h>
#include <faiss/gpu/IndexProxy.h>

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

    std::cout<<"File read done"<<std::endl;
    for(int i =0 ;i< 10;i++){
        for(int j=0 ;j<1024;j++){
            std::cout<<data[i*1024+j]<<" ";
        }
        std::cout<<std::endl;
    }

    // query input
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 14);
    std::string proto_file = "/home/slh/faiss_index/model/deploy_person.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/model.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(10,proto_file.c_str(), proto_weight.c_str(), "loss3/feat_normalize");
    std::cout<<"done extract"<<std::endl;
    for(int i =0 ;i< 10;i++){
        for(int j=0 ;j<1024;j++){
            std::cout<<xq[i*1024+j]<<" ";
        }
        std::cout<<std::endl;
    }
    int d = 1024;
   // 4*sqrt(count)
    int ncentroids = int(4 * sqrt(count));
    faiss::gpu::StandardGpuResources resources;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = 9;

    faiss::gpu::GpuIndexIVFPQ index (
            &resources, d,
            ncentroids, 32, 8,
            faiss::METRIC_L2,config);
    index.verbose = true;
    index.train(count/2, data);
    std::cout<<"done"<<std::endl;

    index.add (count, data);

    { // I/O demo
        const char *outfilename = "/home/slh/faiss_index/index_store/index_person_map_hasdata.faissindex";
        faiss::Index * cpu_index = faiss::gpu::index_gpu_to_cpu (&index);
        write_index (cpu_index, outfilename);
        printf ("done save \n");
        delete cpu_index;
    }

    //faiss::gpu::
    {
        int k = 10;
        int nq = 10;
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        // different from IndexIVF
        index.setNumProbes(128);
        double t0 = elapsed();
        index.search (nq, xq, k, D, I);
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
