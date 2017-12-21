//
// Created by slh on 17-12-4.
//

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <caffe/caffe.hpp>
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/db.hpp"
using namespace std;
bool ECode(std::string FileName){
    // 1000 + 10 byte
    FILE* _f = fopen(FileName.c_str(), "rb");
    // 2G
    char* _tmp =(char *)malloc(sizeof(char)* 1024 * 1024 * 1024 * 2);
    cout<<"nihao"<<endl;
    char* _buf = (char *)malloc(sizeof(char) * 1000);

    long total = 0;
    while (!feof(_f))
    {
        srand((unsigned)time(NULL));
        int size = fread(_buf, sizeof(char), 1000, _f);
        memcpy(_tmp + total, _buf, size * sizeof(char));
        for(int i=0;i<size;i++){
            char code = 100;
            _tmp[i + total] = _tmp[i + total] ^ code;
        }
        total += size;
        if(size < 1000){
            continue;
        }
        for(int i=0;i<50;i++){
            char _s = (char)(rand() % (126 - 0));
            _tmp[total] = _s;
            total ++;
        }
    }
    fclose(_f);
    _f = fopen(FileName.c_str(), "wb");
    fwrite(_tmp, sizeof(char), total, _f);
    fclose(_f);
    delete _tmp;
    delete _buf;
    return true;
}

bool DCode(std::string FileName, std::string FileType){
    // 1000 + 10 byte
    FILE* _f = fopen(FileName.c_str(), "rb");

    char* _tmp =(char *)malloc(sizeof(char)* 1024 * 1024 * 1024 * 2);
    char* _buf = (char *)malloc(sizeof(char) * 1000);
    long total = 0;
    while (!feof(_f))
    {
        srand((unsigned)time(NULL));
        int size = fread(_buf, sizeof(char), 1000, _f);
        memcpy(_tmp + total, _buf, size * sizeof(char));
        for(int i=0;i<size;i++){
            char code = 100;
            _tmp[i + total] = _tmp[i + total] ^ code;
        }
        total += size;
        int skip_size =0;
        if(size == 1000) {
            skip_size = fread(_buf, sizeof(char), 50, _f);
        }else{
            continue;
        }
        if(skip_size != 50){
            std::cout<<"wrong encode"<<std::endl;
        }
    }
    fclose(_f);
    _f = fopen((".tmp"+FileType).c_str(), "wb");
    fwrite(_tmp, sizeof(char), total, _f);
    fclose(_f);
    delete _tmp;
    delete _buf;
    return true;
}
int main(){
    //
    std::string pro = "/home/slh/faiss_index/data/bcar.prototxt";
    std::string model = "/home/slh/faiss_index/data/bcar.caffemodel";
//    std::string pro1 = "/home/slh/faiss_index/data/person.prototxt";
//    std::string model1 = "/home/slh/faiss_index/data/person.caffemodel";
//    ECode(pro);
//    ECode(model);
    ECode(pro);
    ECode(model);

//    DCode(pro, "p");
//    DCode(model, "m");

//   //net work init
//   std::string pretrained_binary_proto(model);
//   std::string proto_model_file(pro);
//   caffe::Net<float>* net(new caffe::Net<float>(proto_model_file, caffe::TEST));
//   net->CopyTrainedLayersFrom(pretrained_binary_proto);

//    return net;
    return 0;

}