//
// Created by slh on 17-10-25.
//

#include "attr_person.h"
#include <fstream>
using namespace std;
using namespace attrOfPerson;

int main(int args, char** argv){
    float *fea;
    float *X_float;
    int *att;

    if(args <2 ){
        return 0;
    }
    string filename = argv[1];
    int count = atoi(argv[2]);

    float* data = new float[count*1024];
    float* data2 = new float[count*1024];
    FILE* fin = fopen(filename.c_str(),"rb");
    fread(data, sizeof(float),count*1024,fin);
    fclose(fin);

    /// transposed
    for(int i=0;i<count;i++){
        for(int k=0;k<1024;k++){
            data2[k*count+i]= data[i*1024+k];
        }
    }
    delete[] data;

    PersonAttr pa("/home/slh/faiss_index/index_store/file/thr.txt");
    fea = pa.readFile("/home/slh/faiss_index/index_store/file/fea.txt",1024,5);

    printf("read thr\n");
    X_float = pa.get_att(fea,900,800,1024,109,count,"123");
    printf("get X\n");
    att = pa.compare_with_thr(X_float,109,count);

    ofstream out("result.txt",std::ios::out);
    for(int i=0;i<count;i++){
        out<<i<<" ";
        for(int j=0;j<109;j++){
            out<<att[j*count+i]<<" ";
        }
        out<<endl;
    }
    out.close();
    delete[] data2;
    return 0;
}