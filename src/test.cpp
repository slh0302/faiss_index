//
// Created by slh on 17-5-10.
//

#include <feature.h>
#include <fstream>
// extract feature
using namespace std;
using namespace feature_index;

int main(int argc,char** argv){
    google::InitGoogleLogging(argv[0]);
    FeatureIndex index = FeatureIndex();
    std::string proto_file = "/home/slh/faiss_index/model/deploy_googlenet_hash.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel";

    if(argc <= 1 ){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 1;
    }
    ofstream out("/home/slh/caffe-ssd/detecter/examples/_temp/file_list",std::ios::out);
    for ( int i = 1;i<argc;i++){
        out<<argv[i]<<" "<<i<<endl;
    }
    out.close();
    float * xq = index.PictureFeatureExtraction(argc - 1 ,proto_file.c_str(), proto_weight.c_str(), "fc_hash/relu");

    for ( int i = 0;i<argc-1;i++){
        for( int j=0 ;j<1024;j++){
            cout<<xq[j+i*1024]<<" ";
        }
        cout<<endl;
        cout<<endl;
    }
    for ( int i = 0;i<argc-1;i++){
        for( int j=0 ;j<1024;j++){
            if(xq[j+i*1024] > 0.01){
                cout<<1<<" ";
            }
            else
                cout<<0<<" ";
        }
        cout<<endl;
        cout<<endl;
    }

    return 0;
}