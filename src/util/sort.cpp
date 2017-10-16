//
// Created by slh on 17-10-11.
//

//
// Created by slh on 17-5-10.
//

#include <feature.h>
#include "boost/algorithm/string.hpp"
#include <fstream>
#include <vector>
#include <string>
// extract feature
using namespace std;
using namespace feature_index;

struct SortTable1 {
    double sum;
    int info;

    bool operator<(const SortTable1 &A) const {
        return sum < A.sum;
    }

    bool operator>(const SortTable1 &A) const {
        return sum > A.sum;
    }
};

SortTable1* result;

int main(int argc,char** argv){
    google::InitGoogleLogging(argv[0]);
    FeatureIndex index = FeatureIndex();
    std::string proto_file = "/home/slh/faiss_index/model/deploy_google_multilabel.prototxt";
    std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";

    if(argc <= 1 ){
        std::cout<<"argc : "<<argc<<" is not enough"<<std::endl;
        return 1;
    }

    string file_list = argv[1];
    ifstream infile(file_list.c_str(), ios::in);
    int count = atoi(argv[2]);
    string save_filename = argv[3];
    Info_String* info = new Info_String[count];
    result = new SortTable1[count];

    FILE * floatRead = fopen(save_filename.c_str(),"rb");
    float* storeIndex = new float[count * 1024];
    fread(storeIndex, sizeof(float), count*1024, floatRead);
    fclose(floatRead);

    floatRead = fopen((save_filename+"_info").c_str(),"rb");
    fread(info, sizeof(Info_String), count, floatRead);
    fclose(floatRead);

    string* s = new string[count];
    std::vector< std::string > file_name_list;
    string temp;
    std::string ROOT_DIR = "/home/slh/retrieval/test_pic/";
    std::string file_list_name ="/home/slh/caffe-ssd/detecter/examples/_temp/file_list";
    std::ofstream output(file_list_name,std::ios::out);

    for(int k =0; k<count; k++){
        getline(infile, temp);
	    boost::split(file_name_list, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
        s[k] = ROOT_DIR + file_name_list[0];
        output<<s[k]<<std::endl;
    }
    infile.close();
    output.close();

    caffe::Net<float>* net = index.InitNet(proto_file, proto_weight);


    float* data ;
    //int count, caffe::Net<float> * _net, std::string blob_name
    data = index.PictureFeatureExtraction(count, net, "pool5/7x7_s1");

    // normalize
    for(int k=0;k<count;k++){
        double result = 0;
        for(int i=0;i<1024;i++){
            double tmp = data[i+k*1024] * data[i+k*1024];
            result += tmp;
        }
        for(int i=0;i<1024;i++){
            data[i+k*1024] = data[i+k*1024] / sqrt(result);
        }
    }

    for(int k=0;k<count;k++){
        double result = 0;
        for(int i=0;i<1024;i++){
            result += data[i+k*1024];
        }
        for(int i=0;i<1024;i++){
            data[i+k*1024] = sqrt(data[i+k*1024]) / sqrt(result);
        }
    }


     for(int ii=0;ii<count;ii++){
        for(int i=0;i<1024;i++){
            std::cout<<storeIndex[i+ii*1024]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }

     for(int ii=0;ii<count;ii++){
        for(int i=0;i<1024;i++){
            std::cout<<data[i+ii*1024]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }



	
    std::ofstream op("/home/slh/result2.txt",std::ios::out);
    for(int k=0;k<count;k++){
        // data[k]
        for(int t =0;t<count;t++){
            result[t].info =t;
            result[t].sum =0;
            for(int i=0;i<1024;i++){
                result[t].sum += (storeIndex[i+t*1024] - data[i+k*1024]) * (storeIndex[i+t*1024] - data[i+k*1024]);
	    	}
	    	std::cout<<t<<" "<<result[t].sum<<std::endl;
        }
        std::sort(result, result + count);
        // name
        op<<s[k]<<std::endl;
        for(int j =0;j<count;j++){
            op<<j<<" "<<info[result[j].info].info<<" "<<result[j].sum<<std::endl;
        }
    }
    op.close();

    delete data;
    return 0;
}
