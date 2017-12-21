//
// Created by slh on 17-11-18.
//
#include <iostream>
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

typedef struct
{
    char ServerIP[MAX_IP_SIZE];//000.000.000.000
    char RelativePath[MAX_PATH_SIZE];
    char FileName[MAX_FILE_NAME_SIZE];
    unsigned int FrameNum;
    unsigned int FrameType;
    unsigned int BoundingBoxNum;
}FeatureMsgInfo;
bool quit_flag = false;

/// Mutex for adding index
pthread_mutex_t mutex_index;
pthread_cond_t index_pause = PTHREAD_COND_INITIALIZER;
const int FEATURE_LENGTH = 128;
const int MAX_LIST_NUM = 10000;
const string IndexFileName = "Video.index";
const string IndexFileNameInfo = "Video.info";
int InfoDataBaseNum = 0;
int FeatureDataBaseNum = 0;
typedef struct
{
    int xlu; /// left up
    int ylu;
    int xrd; /// right down
    int yrd;
    int info_id;
    unsigned char data[FEATURE_LENGTH];
}FeatureWithBox;
vector<FeatureMsgInfo*> _info_list(MAX_LIST_NUM);
vector<FeatureWithBox*> _featrue_list(MAX_LIST_NUM);
/// Index Serialization thread function
int ChangeDataNum(int num, FILE* _f) {
    if (_f == NULL ) {
        return -1;
    }
    fseek(_f, 0, SEEK_SET);
    int has = 0;
    fread(&has, sizeof(int), 1, _f);
    fseek(_f, 0, SEEK_SET);
    int tmp = has + num;
    fwrite(&tmp, sizeof(int), 1, _f);
    fseek(_f, 0, SEEK_END);
    return has;
}
void* IndexSerialization(void*) {
    /// begin process
    pthread_mutex_lock(&mutex_index);
    FILE* _init = fopen(IndexFileNameInfo.c_str(), "ab+");
    int num = ChangeDataNum(0, _init);
    if (num) {
        InfoDataBaseNum = num;
    }
    _init = fopen(IndexFileName.c_str(), "ab+");
    num = ChangeDataNum(0, _init);
    if (num) {
        FeatureDataBaseNum = num;
    }
    fclose(_init);
    pthread_mutex_unlock(&mutex_index);
    /// begin Serialization
    while (1) {
        pthread_mutex_lock(&mutex_index);
        int size = _info_list.size();
        int sizeFeature = _featrue_list.size();
        if(quit_flag && size == 0){
            pthread_mutex_unlock(&mutex_index);
            break;
        }
        if (size == 0) {
            pthread_mutex_unlock(&mutex_index);
            /// wait three minutes
            sleep(10);
            continue;
        }
        // Do Serialization
        FILE* _f = fopen(IndexFileNameInfo.c_str(), "ab+");
        int _re = ChangeDataNum(size, _f);
        if(_re != -1){
            pthread_mutex_unlock(&mutex_index);
            cout << "Error. File Wrong !" << endl;
            continue;
        }
        InfoDataBaseNum += size;
        fwrite(_featrue_list.data(), sizeof(FeatureMsgInfo), size, _f);
        fclose(_f);
        /// Index data
        _f = fopen(IndexFileName.c_str(), "ab+");
        ChangeDataNum(sizeFeature, _f);
        fwrite(_info_list.data(), sizeof(FeatureWithBox), sizeFeature, _f);
        fclose(_f);
        FeatureDataBaseNum += sizeFeature;
        _info_list.erase(_info_list.begin(), _info_list.end());
        _featrue_list.erase(_featrue_list.begin(), _featrue_list.end());
        pthread_mutex_unlock(&mutex_index);
    }
}
pthread_t ntid;

int main(){
//    int err;
//    err = pthread_create(&ntid, NULL, IndexSerialization, NULL);
//    if(err != 0)
//    {
//        printf("can't create new thread \n");
//        exit(-1);
//    }
    for(int k = 0;k<100;k++) {
/// Modify:
///  Adding Index
        pthread_mutex_lock(&mutex_index);
        FeatureMsgInfo *fmi = new FeatureMsgInfo();
        strcpy(fmi->FileName, "89");
        fmi->BoundingBoxNum = 2;
        strcpy(fmi->RelativePath, "./");
        fmi->FrameType = 1;
        strcpy(fmi->ServerIP, "localhost");
/// copy data
        FeatureMsgInfo *_tmp = new FeatureMsgInfo();
        memcpy(_tmp, fmi, sizeof(FeatureMsgInfo));
        int id_feature = _info_list.size();
        _info_list.push_back(_tmp);
        char *featureData = new char[ FEATURE_LENGTH * fmi->BoundingBoxNum  ];
        memset(featureData, 1, sizeof(char) * FEATURE_LENGTH * fmi->BoundingBoxNum);
        cout << "Done one !" << endl;
        for (int i = 0; i < fmi->BoundingBoxNum; i++) {
            FeatureWithBox *_fb = new FeatureWithBox();
            _fb->xlu = 10;
            _fb->ylu = 10;
            _fb->xrd = 20;
            _fb->yrd = 20;
            memcpy(_fb->data, featureData + i * FEATURE_LENGTH,
                   sizeof(char) * FEATURE_LENGTH);
            _fb->info_id = id_feature + InfoDataBaseNum;
            _featrue_list.push_back(_fb);
        }
        cout << "Done one 2 !" << endl;
        pthread_mutex_unlock(&mutex_index);
        delete fmi;
/// Done modify
    }
    quit_flag = true;

    return 1;
}
