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
    std::string proto_file = "/home/slh/faiss_index/model/deploy_person.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/model.caffemodel";

    if(argc <= 1 ){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 1;
    }

    int count = atoi(argv[2]);
    string file_prefix = argv[1];
    ifstream file_list("/home/slh/data/peta/list/" + file_prefix + ".list", ios::in);
    string* info_array = new string[count];
    string name = "";
    string ROOT_DIR = "/home/slh/data/peta/" + file_prefix + "/archive/";
    cout << ROOT_DIR << endl;
    ofstream out("/home/slh/caffe-ssd/detecter/examples/_temp/file_list", std::ios::out);
    for ( int i = 0;i<count;i++){
        file_list >> name;
        info_array[i] = name;
        out<<ROOT_DIR+name<<" "<<i<<endl;
    }
    out.close();
    file_list.close();
    index.InitGpu("GPU", 11);
    float * xq = index.PictureFeatureExtraction(count ,proto_file.c_str(), proto_weight.c_str(), "loss3/feat_normalize");

    string result_file = file_prefix + ".out";
    ofstream result(result_file, ios::out);
    for ( int i = 0;i<count;i++){
        result<< info_array[i] << " ";
        for( int j=0 ;j<1024;j++){
            result<<xq[j+i*1024]<<" ";
        }
        result<<endl;
    }
    result.close();

    return 0;
}