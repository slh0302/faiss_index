//
// Created by slh on 17-11-9.
//

#include<iostream>
#include<cstdio>
#include<cstdlib>
using namespace std;
struct Info_String
{
    char info[100];
};

int main(int argc ,char** argv){

    FILE* f=fopen(argv[2],"rb");
    FILE* f2= fopen(argv[1],"rb");
    int count = atoi(argv[3]);
    float* data = new float[count*1024];
    Info_String* info =new Info_String[atoi(argv[3])];
    fread(info,sizeof(Info_String),count, f);
    fread(data,sizeof(float),count*1024, f2);
    fclose(f);
    fclose(f2);
    for(int j =0;j<1024;j++){
        cout<<data[j]<<" ";
    }
    cout<<endl;
    FILE* o = fopen("test.out","wb");
    fwrite(data, sizeof(float), 1024, o);
    fclose(o);
    return 0;
}