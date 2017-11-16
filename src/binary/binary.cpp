//
// Created by slh on 2017/8/26.
//

#include "binary.h"



#include "cstring"
#include "boost/timer.hpp"
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



int FeatureBinary::IndexTable[( 1<< BYTE_INDEX )];
std::map<int, int> FeatureBinary::LabelList;


// Basic Function
void FeatureBinary::swap(FeatureBinary::SortTable *a, FeatureBinary::SortTable *b)
{
    int tmp;
    tmp = a->sum;
    a->sum = b->sum;
    b->sum = tmp;
    int trans;
    trans = a->info;
    a->info = b->info;
    b->info = trans;
}


int FeatureBinary::partition(FeatureBinary::SortTable arr[], int left, int right, int pivotIndex){
    int storeIndex = left;
    int pivotValue = arr[pivotIndex].sum;
    int i;

    FeatureBinary::swap(&arr[pivotIndex], &arr[right]);

    for (i = left; i < right; i++)
    {
        if (arr[i].sum <= pivotValue)
        {
            FeatureBinary::swap(&arr[i], &arr[storeIndex]);
            storeIndex++;
        }
    }
    FeatureBinary::swap(&arr[storeIndex], &arr[right]);
    return storeIndex;
}


int FeatureBinary::findKMax(FeatureBinary::SortTable arr[], int left, int right, int k){
    int nRet;
    int pivotIndex = left;

    nRet = partition(arr, left, right, pivotIndex);
    if (nRet < k)
    {
        return FeatureBinary::findKMax(arr, nRet + 1, right, k);
    }
    else if (nRet > k)
    {
        return FeatureBinary::findKMax(arr, left, nRet - 1, k);
    }
    return nRet;
}


void FeatureBinary::InitIndex(void* p, unsigned char* filename, FeatureBinary::Info_String* in_str,int count){
//
    FeatureBinary::feature* temp = (FeatureBinary::feature*)p;
    int i = 0;
    FeatureBinary::DataSet* s = new FeatureBinary::DataSet[count];
    while (i<count){
        for (int j = 0; j < TOTALBYTESIZE / BYTE_INDEX; j++){
            unsigned int y = 0;
            y = filename[i*TOTALBYTESIZE / 8 + j* BYTE_INDEX / 8 ];
            y = y << 8;
            y = y | filename[i*TOTALBYTESIZE / 8 + j* BYTE_INDEX / 8 + 1];
            s[i].data[j] = y;
        }
        i++;
    }
    temp->setDataSet(s,count);
    temp->setInfo(in_str, count);
}

int* FeatureBinary::DoHandle(unsigned char * dat){
    int * data = new int[TOTALBYTESIZE / BYTE_INDEX];
    for (int j = 0; j < TOTALBYTESIZE / BYTE_INDEX; j++){
        unsigned int y = 0;
        y = dat[j* BYTE_INDEX / 8 ];
        y = y << 8;
        y = y | dat[j* BYTE_INDEX / 8 + 1];
        data[j] = y;
    }
    return data;
}


void* FeatureBinary::CreateIndex(int size){
    FeatureBinary::feature* temp = new FeatureBinary::feature;
    return (void*)temp;
}


void FeatureBinary::CreateTable(const char * filename,int bits){
    std::ifstream ifstream_in(filename, std::ios::in);
    int x, y;
    for (int i = 0; i < (1 << bits); i++){
        ifstream_in >> x >> y;
        FeatureBinary::IndexTable[x]=y;
    }
    ifstream_in.clear();
    ifstream_in.close();
}


bool FeatureBinary::AddToIndex(void*p, int* data, const char* in){
    FeatureBinary::feature* temp = new FeatureBinary::feature;
    int total = temp->getTotalSize();
    int curNum = temp->getCount();
    if (total < curNum+1){
        FeatureBinary::DataSet* ne = new FeatureBinary::DataSet[curNum * 2];
        FeatureBinary::Info_String* info_ne = new FeatureBinary::Info_String[curNum * 2];
        memcpy(ne, temp->getDataSet(), sizeof(FeatureBinary::DataSet)*curNum);
        memcpy(info_ne, temp->getInfoSet(), sizeof(FeatureBinary::Info_String)*curNum);
        memcpy(ne[curNum].data,  data, sizeof(int[64]));
        strcpy(info_ne[curNum].info, in);
        temp->deleteData();
        temp->setDataSet(ne, curNum+1);
        temp->setInfo(info_ne, curNum+1);
        temp->setTotalSize(2 * curNum);
    }
    else{
        memcpy(&(temp->getDataSet()[curNum]), data, sizeof(int[64]));
        memcpy(&(temp->getInfoSet()[curNum]),in,  sizeof(FeatureBinary::Info_String));
        temp->setCount(curNum + 1);
    }
    return true;
}


bool FeatureBinary::LoadIndex(void* p, const char* filename,const char* info_file, int count){
    FILE* in = fopen(filename, "rb");
    FILE* in_info = fopen(info_file, "rb");
    FeatureBinary::feature* temp = (FeatureBinary::feature*)p;
    FeatureBinary::DataSet* da = new FeatureBinary::DataSet[count];
    FeatureBinary::Info_String* inst = new FeatureBinary::Info_String[count];

    fread(da, sizeof(DataSet) , count, in);
    fread(inst, sizeof(Info_String), count, in_info);

    fclose(in);
    fclose(in_info);
    temp->setInfo(inst, count);
    temp->setDataSet(da, count);
    return true;
}

bool FeatureBinary::Load_SpData(FeatureBinary::DataSet*da, FeatureBinary::Info_String* inst, const char* filename,const char* info_file, int count,int sp_begin){
    FILE* in = fopen(filename, "rb");
    FILE* in_info = fopen(info_file, "rb");
    fread(&da[sp_begin], sizeof(FeatureBinary::DataSet) , count, in);
    fread(&inst[sp_begin], sizeof(FeatureBinary::Info_String), count, in_info);
    fclose(in);
    fclose(in_info);
    return true;
}

bool FeatureBinary::DeleteIndex(void* p){
    FeatureBinary::feature* temp = new FeatureBinary::feature;
    delete[] temp;
    return true;
}


bool FeatureBinary::ArchiveIndex(void* p, const char* filename, const char* info_file,int count, char mode){
    FeatureBinary::feature* temp = new FeatureBinary::feature;
    FILE* os, *os_info;
    if (mode == 'a'){
        os = fopen(filename, "ab");
        os_info = fopen(info_file, "ab");

    }
    else{
        os = fopen(filename, "wb");
        os_info = fopen(info_file, "wb");
    }
    fwrite(temp->getInfoSet(), sizeof(Info_String),count , os_info);
    fwrite(temp->getDataSet(), sizeof(DataSet), count, os);
    fclose(os);
    fclose(os_info);
    return true;
}

int FeatureBinary::retrival(int* input, FeatureBinary::DataSet* get_t, FeatureBinary::Info_String* get_info,
                            int total,string& result,int bits,int LIMIT, FeatureBinary::SortTable* sorttable){
    int calc = 0;
    int i = 0, indexLine = 0;
    int sum = 0;
    unsigned short Record[TOTALBYTESIZE / BYTE_INDEX];
    std::cout<<"start"<<std::endl;
    while (calc < total){
        sum = 0;
        memset(Record, 0, sizeof(Record));
        for (i = 0; i < TOTALBYTESIZE / bits; i++){
            Record[i] = input[i] ^get_t[calc].data[i];
            sum += FeatureBinary::IndexTable[Record[i]];
            if (sum > LIMIT){
                break;
            }
        }
        if (sum < LIMIT){
            sorttable[indexLine].sum = sum;
            sorttable[indexLine++].info = calc;
            calc ++;
        }
        else{
            calc++;
        }

    }
    std::cout<<"done "<<indexLine<<std::endl;
    int num_find=300<indexLine?300:indexLine;
    std::cout<<"num_find "<<num_find<<std::endl;
    FeatureBinary::findKMax(sorttable, 0, indexLine - 1,num_find);
    std::sort(sorttable, sorttable + num_find-1);
    result = get_info[sorttable[0].info].info;
    cout<<"done sort"<<endl;
    return num_find;
}
