//
// Created by slh on 17-5-11.
//

#include  <feature.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
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
    // file read data
    FILE* f = fopen (argv[1],"rb");
    if(f == NULL){
        std::cout<<"File "<<argv[1]<<" is not right"<<std::endl;
        return 0;
    }
    fread(data,sizeof(float), count*1024, f);
    fclose(f);

    // file read info
    std::string file_info = std::string(argv[1])+"_info";
    FILE* f2 = fopen(file_info.c_str(),"rb");
    feature_index::Info_String* info = new feature_index::Info_String[count];
    fread(info, sizeof(feature_index::Info_String), count, f2);
    fclose(f2);
    std::cout<<"File read done"<<std::endl;



    // query input
    feature_index::FeatureIndex input_index;
    input_index.InitGpu("GPU", 0);
    std::string proto_file = "/home/slh/faiss_index/model/deploy_google_multilabel.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";
    float * xq = input_index.PictureFeatureExtraction(10, proto_file.c_str(), proto_weight.c_str(), "pool5/7x7_s1");
    std::cout<<"done extract"<<std::endl;

    // query info input
    int queryNum = 10;
    int* query = new int[queryNum];
    std::ifstream inquery("/home/slh/faiss_index/model/queryinfo",std::ios::in);
    for(int i=0; i<queryNum ;i++){
        inquery>>query[i];
    }
    inquery.close();


    // Init label list
    input_index.InitLabelList("/home/slh/faiss_index/model/labellist.txt");

    int d = 1024;
    int ncentroids = int(4 * sqrt(count));
    faiss::gpu::StandardGpuResources resources;
    //    GpuIndexIVFFlat(GpuResources* resources,
    //                    int device,
    //            // Does the coarse quantizer use float16?
    //            bool useFloat16CoarseQuantizer,
    //            // Is our IVF storage of vectors in float16?
    //            bool useFloat16IVFStorage,
    //            int dims,
    //            int nlist,
    //            IndicesOptions indicesOptions,
    //            faiss::MetricType metric);
    //    faiss::gpu::GpuIndexFlat quantizer(&resources,d, faiss::METRIC_L2);
    faiss::gpu::GpuIndexIVFFlat index (&resources, 1,false,false,d,ncentroids,
                                       faiss::gpu::INDICES_64_BIT, faiss::METRIC_L2);

    index.verbose = true;
    //resources.setTempMemory(512 * 1024 * 1024);
    index.train(count, data);
    std::cout<<"done extract"<<std::endl;
    //for(int i=0;i<2;i++){
    index.add(count, data);
    // }


//    { // I/O demo
//        const char *outfilename = "/home/slh/faiss_index/index_store/index_car_IVF.faissindex";
//        faiss::Index * cpu_index = faiss::gpu::index_gpu_to_cpu (&index);
//        write_index (cpu_index, outfilename);
//        printf ("done save \n");
//        delete cpu_index;
//    }

    {
        int k = 20;
        int nq = 10;
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        // different from IndexIVF
        index.setNumProbes(10);
        nq = 10;
        double t0 = elapsed();
        index.search (nq, xq, k, D, I);
        double t1 = elapsed();

        //evaluate
        double total_res = 0;
        for(int i = 0; i< nq; i++){
            double res = input_index.Evaluate(k, query[i], info, &I[i * k]);
            total_res += res;
        }
        printf("mAP:  %7lf\n", total_res/nq);


        printf("%7lf \n",t1 - t0 );
        printf("I=\n");
        for(int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%7ld ", I[i * k + j]);
            printf("\n");
        }
        printf("D=\n");
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                printf ("%7g ", D[j + i * k]);
            }
            printf ("\n");
        }

    }
    return 0;

}
