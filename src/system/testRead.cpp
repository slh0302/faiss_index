//
// Created by slh on 17-11-22.
//

//
// Created by slh on 17-11-18.
//
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <pthread.h>

using namespace std;
/// other pre-define type or value
# define MAX_IP_SIZE			16
# define MAX_PATH_SIZE			32
# define MAX_FILE_NAME_SIZE		32
# define MAX_PKG_SIZE			40000

const int FEATURE_LENGTH = 128;
typedef struct
{
    char ServerIP[MAX_IP_SIZE];//000.000.000.000
    char RelativePath[MAX_PATH_SIZE];
    char FileName[MAX_FILE_NAME_SIZE];
    unsigned int FrameNum;
    unsigned int FrameType;
    unsigned int BoundingBoxNum;
}FeatureMsgInfo;

typedef struct
{
    int xlu; /// left up
    int ylu;
    int xrd; /// right down
    int yrd;
    int info_id;
    char data[FEATURE_LENGTH];
}FeatureWithBox;

const string IndexFileName = "Video.index";
const string IndexFileNameInfo = "Video.info";

int main(){
    FILE* _f = fopen(IndexFileName.c_str(), "rb");
    FILE* _f1 = fopen(IndexFileNameInfo.c_str(), "rb");
    int num_data = 0;
    int data_info = 0;
    fread(&num_data, sizeof(int), 1, _f);
    fread(&data_info, sizeof(int), 1, _f1);
    cout<< num_data << " "<< data_info <<endl;
    FeatureMsgInfo* s = new FeatureMsgInfo[data_info];
    FeatureWithBox* tmp = new FeatureWithBox[num_data];
    fseek(_f, sizeof(int), SEEK_SET);
    fseek(_f1, sizeof(int), SEEK_SET);
    fread(s, sizeof(FeatureMsgInfo), data_info, _f1);
    fread(tmp, sizeof(FeatureWithBox), num_data, _f);
    fclose(_f);
    fclose(_f1);
    for( int k=0;k< num_data;k++){
        cout<< k<<" "<<s[tmp[k].info_id].FileName <<" "<< tmp[k].yrd <<endl;
    }
    cout<< num_data << " "<< data_info <<endl;
    return 0;
}